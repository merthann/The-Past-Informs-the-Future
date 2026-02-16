import sys
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soccerdata as sd
except ImportError:
    print("soccerdata not installed")
    sys.exit(1)


script_dir = Path(__file__).parent

# Match ID dosyası
match_ids_file = script_dir / "match_ids_championship_1671.json"
if not match_ids_file.exists():
    print(f"❌ Championship Match IDs file not found: {match_ids_file}")
    print("   Please run collect_championship_match_ids.py first")
    sys.exit(1)

print(f"📁 Using match IDs file: {match_ids_file.name}")

with open(match_ids_file, 'r') as f:
    all_matches = json.load(f)

# Position label encoding derived from WhoScored position tags
POSITION_ENCODING = {
    "GK": 0,
    "DC": 1,
    "DL": 2,
    "DR": 3,
    "DMC": 4,
    "DML": 5,
    "DMR": 6,
    "MC": 7,
    "ML": 8,
    "MR": 9,
    "AMC": 10,
    "AML": 11,
    "AMR": 12,
    "FW": 13,
    "FWL": 14,
    "FWR": 15,
}
UNKNOWN_POSITION_CODE = max(POSITION_ENCODING.values()) + 1

print("=" * 60)
print(f"Processing {len(all_matches)} Championship Matches")
print("Processing for 5 time periods: 30, 45, 60, 75, 90 minutes")
print("=" * 60)

# 5 zaman dilimi için işleme yapılacak
TIME_PERIODS = [30, 45, 60, 75, 90]
# Eğer 45, 60, 75, 90 varsa sadece 30'u çekmek için kontrol edilecek zaman dilimleri
EXISTING_PERIODS_TO_CHECK = [45, 60, 75, 90]

# Ana data klasörü (parent directory)
base_data_dir = script_dir.parent / "data"
base_data_dir.mkdir(parents=True, exist_ok=True)

# Her maç için beklenen CSV dosyaları
REQUIRED_FILENAMES = [
    "passes.csv",
    "features.csv",
    "events.csv",
    "players.csv",
    "player_positions.csv",
    "node_features.csv",
    "player_event_counts.csv",
    "player_event_summary.csv",
]

def create_scraper():
    return sd.WhoScored(headless=False)

def safe_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.\-]", "", value)
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def get_display_value(value):
    if isinstance(value, dict):
        return value.get("displayName") or value.get("name") or value.get("value")
    return value


def encode_position_label(position: str) -> int:
    """
    Convert WhoScored position tags into deterministic label-encoded values.
    Falls back to an 'unknown' bucket for unseen tags to keep consistency.
    """
    if not position or not isinstance(position, str):
        return UNKNOWN_POSITION_CODE
    pos = position.strip().upper()
    return POSITION_ENCODING.get(pos, UNKNOWN_POSITION_CODE)


def sanitize_event_column(name: str) -> str:
    label = str(name) if name is not None else "unknown"
    label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
    return f"event_{label}" if label else "event_unknown"


def extract_player_rating(player_data, stats=None):
    """WhoScored'dan rating verisini çekmek için farklı yolları dene"""
    rating = None
    
    # Yöntem 1: stats.ratings (en yaygın yöntem - dakika bazında rating'ler)
    if rating is None and stats:
        ratings = stats.get("ratings")
        if ratings is not None:
            if isinstance(ratings, dict):
                # ratings bir dict ve key'ler dakika numaraları (string), değerler rating'ler
                # Örnek: {"0":6.00, "1":6.01, "90":7.50}
                # Son dakikadaki rating'i al (maç sonu rating'i)
                rating_values = []
                for key, val in ratings.items():
                    if isinstance(val, (int, float)):
                        rating_values.append((int(key) if key.isdigit() else 0, float(val)))
                
                if rating_values:
                    # En yüksek dakika numarasındaki rating'i al (maç sonu rating'i)
                    rating_values.sort(key=lambda x: x[0])
                    rating = rating_values[-1][1]  # Son dakikadaki rating
                    # Alternatif: Tüm rating'lerin ortalaması
                    # rating = sum(r[1] for r in rating_values) / len(rating_values)
                else:
                    # Eğer numeric değer yoksa, 'value' veya 'rating' key'ini dene
                    rating = ratings.get("value") or ratings.get("rating") or ratings.get("overall")
            elif isinstance(ratings, (int, float)):
                rating = ratings
    
    # Yöntem 2: Direkt player.rating
    if rating is None:
        rating = player_data.get("rating")
        if isinstance(rating, dict):
            rating = rating.get("value") or rating.get("displayName")
    
    # Yöntem 3: player.stats.rating (tekil)
    if rating is None and stats:
        rating_stat = stats.get("rating")
        if isinstance(rating_stat, dict):
            rating = rating_stat.get("value") or rating_stat.get("displayName")
        elif rating_stat is not None:
            rating = rating_stat
    
    # Yöntem 4: player.ratings (çoğul, direkt player'da)
    if rating is None:
        ratings = player_data.get("ratings")
        if isinstance(ratings, dict):
            rating = ratings.get("value") or ratings.get("overall") or ratings.get("rating")
        elif isinstance(ratings, (int, float)):
            rating = ratings
    
    # Yöntem 5: player.performanceRating veya player.matchRating
    if rating is None:
        rating = player_data.get("performanceRating") or player_data.get("matchRating")
        if isinstance(rating, dict):
            rating = rating.get("value")
    
    # Yöntem 6: stats içinde farklı isimlerle
    if rating is None and stats:
        rating = stats.get("performanceRating") or stats.get("matchRating") or stats.get("overallRating")
        if isinstance(rating, dict):
            rating = rating.get("value")
    
    return safe_float(rating)


def extract_player_stats_from_dict(stats_dict):
    """Player stats dictionary'sinden istatistikleri çıkarır (dakika bazlı dict'lerden toplam değerleri hesaplar)"""
    if not stats_dict or not isinstance(stats_dict, dict):
        return {}
    
    result = {}
    
    # WhoScored'da stats genellikle dakika bazlı dict'ler olarak gelir: {"0": 1, "15": 2, "90": 3}
    # Toplam değeri hesaplamak için tüm değerleri toplarız
    # WhoScored JSON'unda mevcut olan stats key'leri
    stat_keys_to_extract = [
        # Aerials
        'aerialsWon', 'aerialsTotal',
        # Shots
        'shotsTotal', 'shotsOnTarget', 'shotsOffTarget', 'shotsBlocked',
        # Passes
        'passesAccurate', 'passesTotal', 'passSuccess',
        # Corners
        'cornersTotal', 'cornersAccurate',
        # Dribbles
        'dribblesWon', 'dribblesAttempted',
        # Defense
        'clearances', 'tacklesTotal', 'tackleSuccess', 'interceptions',
        # Other
        'saves', 'totalSaves', 'touches', 'dispossessed',
        'foulsCommited', 'possession',
    ]
    
    for key in stat_keys_to_extract:
        value = stats_dict.get(key)
        if value is None:
            result[key] = 0
        elif isinstance(value, dict):
            # Dakika bazlı dict ise, tüm değerleri topla
            total = 0
            for minute_key, minute_value in value.items():
                if isinstance(minute_value, (int, float)):
                    total += minute_value
            result[key] = total
        elif isinstance(value, (int, float)):
            result[key] = value
        else:
            result[key] = 0
    
    return result


def get_captain_id_from_formations(team_info):
    """Formations içinden captain player ID'sini çeker"""
    formations = team_info.get("formations", [])
    if formations:
        # İlk formation'dan (maç başlangıcı) captain ID'sini al
        first_formation = formations[0]
        return safe_int(first_formation.get("captainPlayerId"))
    return None


def get_starting_eleven_ids(team_info):
    """Starting eleven oyuncularının ID'lerini döndürür"""
    starting_ids = set()
    for player in team_info.get("players", []):
        if player.get("isFirstEleven", False):
            player_id = safe_int(player.get("playerId"))
            if player_id:
                starting_ids.add(player_id)
    return starting_ids


def extract_player_rows(match_id, team_side, team_info):
    """
    Player verilerini JSON'dan çeker.
    Captain bilgisi formations içinden alınır.
    """
    rows = []
    if not team_info:
        return rows
    
    team_id = safe_int(team_info.get("teamId"))
    team_name = team_info.get("name", team_side)
    captain_id = get_captain_id_from_formations(team_info)
    
    for player in team_info.get("players", []):
        player_id = safe_int(player.get("playerId"))
        if player_id is None:
            continue
        stats = player.get("stats", {}) or {}
        rating = extract_player_rating(player, stats)
        player_stats = extract_player_stats_from_dict(stats)
        
        rows.append(
            {
                "match_id": match_id,
                "team_side": team_side,
                "team_id": team_id,
                "team_name": team_name,
                "player_id": player_id,
                "player_name": player.get("name", ""),
                "shirt_no": safe_int(player.get("shirtNo")),
                "position": get_display_value(player.get("position")),
                "height_cm": safe_float(player.get("height")),
                "weight_kg": safe_float(player.get("weight")),
                "age": safe_float(player.get("age")),
                "is_first_eleven": bool(player.get("isFirstEleven", False)),
                "is_captain": (player_id == captain_id) if captain_id else False,
                "is_man_of_the_match": bool(player.get("isManOfTheMatch", False)),
                "rating": rating,
                # Stats from WhoScored JSON:
                # Aerials
                "aerials_won": player_stats.get('aerialsWon', 0),
                "aerials_total": player_stats.get('aerialsTotal', 0),
                # Shots
                "shots_total": player_stats.get('shotsTotal', 0),
                "shots_on_target": player_stats.get('shotsOnTarget', 0),
                "shots_off_target": player_stats.get('shotsOffTarget', 0),
                "shots_blocked": player_stats.get('shotsBlocked', 0),
                # Corners
                "corners_total": player_stats.get('cornersTotal', 0),
                "corners_accurate": player_stats.get('cornersAccurate', 0),
                # Dribbles
                "dribbles_won": player_stats.get('dribblesWon', 0),
                "dribbles_attempted": player_stats.get('dribblesAttempted', 0),
                # Defense
                "clearances": player_stats.get('clearances', 0),
                "tackles": player_stats.get('tacklesTotal', 0),
                "tackle_success": player_stats.get('tackleSuccess', 0),
                "interceptions": player_stats.get('interceptions', 0),
                # Other
                "saves": player_stats.get('saves', 0) or player_stats.get('totalSaves', 0),
                "touches": player_stats.get('touches', 0),
                "dispossessed": player_stats.get('dispossessed', 0),
                "fouls_commited": player_stats.get('foulsCommited', 0),
            }
        )
    return rows

def get_match_data_with_retry(match_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            ws = create_scraper()
            match_url = f"https://www.whoscored.com/Matches/{match_id}/Live"
            filepath = ws.data_dir / "events" / f"direct_{match_id}.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            reader = ws.get(
                match_url,
                filepath,
                var="require.config.params['args'].matchCentreData",
                no_cache=False,
            )
            try:
                if hasattr(ws, '_driver') and ws._driver:
                    ws._driver.quit()
            except Exception:
                pass
            return reader
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
            try:
                if 'ws' in locals() and hasattr(ws, '_driver') and ws._driver:
                    ws._driver.quit()
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(1)  # Kısa bekleme
            else:
                print("   ❌ All retries failed")
                return None
    return None

def add_receiver_columns(df_events, player_name_dict=None):
    df_events = df_events.copy()
    df_events['receiver_id'] = pd.NA
    df_events['receiver_name'] = pd.NA
    df_events['possession_retained'] = False
    next_player = df_events['playerId'].shift(-1)
    next_team = df_events['teamId'].shift(-1)
    same_team = (df_events['teamId'] == next_team) & next_player.notna()
    df_events.loc[same_team, 'possession_retained'] = True
    pass_condition = same_team & (df_events['type_name'] == 'Pass')
    df_events.loc[pass_condition, 'receiver_id'] = next_player[pass_condition]
    df_events['receiver_id'] = df_events['receiver_id'].apply(safe_int)
    if player_name_dict:
        df_events['receiver_name'] = df_events['receiver_id'].apply(
            lambda pid: player_name_dict.get(pid) if pid in player_name_dict else None
        )
    return df_events

def is_successful(outcome):
    if isinstance(outcome, dict):
        return outcome.get('value') == 1
    if isinstance(outcome, str):
        return 'Successful' in outcome
    return False


def build_player_event_tables(match_id, df_events):
    work = df_events.copy()
    work['match_id'] = match_id
    work['team_id'] = work['teamId'].apply(safe_int) if 'teamId' in work.columns else pd.Series(index=work.index, dtype='Int64')
    work['player_id'] = work['playerId'].apply(safe_int) if 'playerId' in work.columns else pd.Series(index=work.index, dtype='Int64')
    if 'team_name' in work.columns:
        work['team_name'] = work['team_name'].fillna('')
    else:
        work['team_name'] = pd.Series('', index=work.index, dtype=object)
    if 'player_name' in work.columns:
        work['player_name'] = work['player_name'].fillna('')
    else:
        work['player_name'] = pd.Series('', index=work.index, dtype=object)
    if 'type_name' in work.columns:
        work['event_type'] = work['type_name'].fillna('UnknownEvent')
    else:
        work['event_type'] = pd.Series('UnknownEvent', index=work.index, dtype=object)
    work = work.dropna(subset=['player_id'])
    if work.empty:
        columns = ['match_id', 'team_id', 'team_name', 'player_id', 'player_name', 'event_type', 'event_count']
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns[:-2]), defaultdict(Counter)
    grouped = (
        work.groupby(['match_id', 'team_id', 'team_name', 'player_id', 'player_name', 'event_type'], dropna=False)
        .size()
        .reset_index(name='event_count')
    )
    pivot = (
        grouped.pivot_table(
            index=['match_id', 'team_id', 'team_name', 'player_id', 'player_name'],
            columns='event_type',
            values='event_count',
            fill_value=0,
        )
        .reset_index()
    )
    renamed_cols = []
    for col in pivot.columns:
        if col in ['match_id', 'team_id', 'team_name', 'player_id', 'player_name']:
            renamed_cols.append(col)
        else:
            if isinstance(col, tuple):
                event_label = col[-1]
            else:
                event_label = col
            renamed_cols.append(sanitize_event_column(event_label))
    pivot.columns = renamed_cols
    player_event_counter = defaultdict(Counter)
    for _, row in grouped.iterrows():
        pid = row['player_id']
        if pd.isna(pid):
            continue
        player_event_counter[int(pid)][row['event_type']] = row['event_count']
    return grouped, pivot, player_event_counter


def calculate_player_positions(df_events, team_id, starting_eleven_ids=None, time_cutoff=90):
    """
    Event'lerden her oyuncu için average X ve Y koordinatlarını hesaplar.
    PDF'te belirtildiği gibi: "For the X, Y coordinates, we take the average of the 
    coordinates recorded for each player up to the prediction timeframe."
    
    Sadece starting eleven oyuncuları için hesaplama yapar.
    time_cutoff: Hangi dakikaya kadar olan eventler dahil edilecek (30, 45, 60, 75, 90)
    """
    if df_events.empty:
        return []
    
    # Zaman dilimine göre event'leri filtrele
    time_col = 'expandedMinute' if 'expandedMinute' in df_events.columns else 'minute' if 'minute' in df_events.columns else None
    if time_col:
        df_events = df_events[df_events[time_col] <= time_cutoff].copy()
    
    # Takıma ait event'leri filtrele
    team_events = df_events[df_events['teamId'] == team_id].copy()
    if team_events.empty:
        return []
    
    # x ve y koordinatları olan event'leri filtrele
    team_events = team_events.dropna(subset=['x', 'y'])
    team_events = team_events[team_events['x'] != 0]  # Start event'lerini hariç tut
    
    if team_events.empty:
        return []
    
    # Her oyuncu için ortalama koordinatları hesapla
    positions = []
    player_groups = team_events.groupby('playerId')
    
    for player_id, group in player_groups:
        player_id_int = safe_int(player_id)
        if player_id_int is None:
            continue
        
        # Sadece starting eleven oyuncuları için hesapla (eğer liste verilmişse)
        if starting_eleven_ids is not None and player_id_int not in starting_eleven_ids:
            continue
        
        avg_x = group['x'].mean()
        avg_y = group['y'].mean()
        
        # Oyuncu adını al
        player_name = ''
        if 'player_name' in group.columns and not group['player_name'].isna().all():
            player_name = group['player_name'].iloc[0]
        
        positions.append({
            'player_id': player_id_int,
            'player_name': player_name,
            'avg_x': float(avg_x),
            'avg_y': float(avg_y),
            'event_count': len(group),
        })
    
    return positions


def append_player_positions(match_id, team_label, team_name, team_id, players):
    """Player pozisyonlarını DataFrame satırlarına dönüştürür"""
    rows = []
    for player in players:
        rows.append({
            'match_id': match_id,
            'team': team_label,
            'team_name': team_name,
            'team_id': team_id,
            'player_id': player['player_id'],
            'player_name': player['player_name'],
            'avg_x': player['avg_x'],
            'avg_y': player['avg_y'],
        })
    return rows


def build_node_features(match_id, positions_df, passes_df, player_info_df, player_event_counter):
    """
    PDF Table 3'teki node features'ları oluşturur:
    - Statik: position, height, weight
    - Dinamik: rating, pass accuracy, avg_x, avg_y
    """
    if positions_df.empty:
        return pd.DataFrame()
    
    info_lookup = {}
    required_info_cols = ['team_id', 'player_id', 'position', 'shirt_no', 'rating', 
                          'height_cm', 'weight_kg', 'is_first_eleven', 'is_captain']
    stats_cols = ['aerials_won', 'aerials_total', 'shots_total', 'shots_on_target',
                  'corners_total', 'dribbles_won', 'clearances', 'tackles', 'tackle_success',
                  'saves', 'touches', 'dispossessed', 'goals', 'assists']
    all_info_cols = required_info_cols + [col for col in stats_cols if col in player_info_df.columns]
    
    if not player_info_df.empty:
        available_cols = [col for col in all_info_cols if col in player_info_df.columns]
        if 'team_id' in available_cols and 'player_id' in available_cols:
            info_lookup = (
                player_info_df[available_cols]
                .drop_duplicates(['team_id', 'player_id'])
                .set_index(['team_id', 'player_id'])
                .to_dict('index')
            )
    
    # Pass istatistiklerini hesapla
    passes_work = passes_df.copy() if not passes_df.empty else pd.DataFrame()
    if not passes_work.empty:
        passes_work['player_id'] = passes_work['playerId'].apply(safe_int)
        passes_work['receiver_id'] = passes_work.get(
            'receiver_id',
            pd.Series(index=passes_work.index, dtype='Int64')
        ).apply(safe_int)
        passes_work = passes_work.dropna(subset=['player_id'])
        
        success_series = passes_work.get('possession_retained')
        if success_series is None:
            success_series = passes_work['outcomeType'].apply(is_successful)
        success_series = success_series.fillna(False).astype(bool)
        
        pass_counts = passes_work.groupby('player_id').size().to_dict()
        pass_success_counts = passes_work[success_series].groupby('player_id').size().to_dict()
        received_counts = (
            passes_work.dropna(subset=['receiver_id']).groupby('receiver_id').size().to_dict()
            if 'receiver_id' in passes_work.columns
            else {}
        )
    else:
        pass_counts = {}
        pass_success_counts = {}
        received_counts = {}
    
    node_rows = []
    for _, player in positions_df.iterrows():
        player_id = safe_int(player.get('player_id'))
        team_id = safe_int(player.get('team_id'))
        if player_id is None:
            continue
        
        info = info_lookup.get((team_id, player_id), {})
        
        # Pass accuracy hesapla (PDF'te belirtildiği gibi)
        passes_made = pass_counts.get(player_id, 0)
        passes_completed = pass_success_counts.get(player_id, 0)
        pass_accuracy = passes_completed / passes_made if passes_made else 0.0
        passes_received = received_counts.get(player_id, 0)
        
        event_counts = player_event_counter.get(player_id, Counter())
        
        raw_position = info.get('position')
        if raw_position is None and not player_info_df.empty:
            fallback = player_info_df[player_info_df['player_id'] == player_id]
            if not fallback.empty:
                raw_position = fallback.iloc[0].get('position')
        position_code = encode_position_label(raw_position)
        
        node_rows.append({
            'match_id': match_id,
            'team': player.get('team'),
            'team_id': team_id,
            'team_name': player.get('team_name'),
            'player_id': player_id,
            'player_name': player.get('player_name'),
            'position_encoded': position_code,
            # Statik özellikler (PDF Table 3)
            'position': info.get('position'),
            'shirt_no': info.get('shirt_no'),
            'height_cm': info.get('height_cm'),
            'weight_kg': info.get('weight_kg'),
            'is_first_eleven': info.get('is_first_eleven'),
            'is_captain': info.get('is_captain', False),
            # Dinamik özellikler (PDF Table 3)
            'rating': info.get('rating'),
            'avg_x': player.get('avg_x'),
            'avg_y': player.get('avg_y'),
            'passes_made': passes_made,
            'passes_completed': passes_completed,
            'passes_received': passes_received,
            'pass_accuracy': pass_accuracy,
            # Event bazlı istatistikler
            'ball_touches': event_counts.get('BallTouch', 0) or info.get('touches', 0),
            'take_ons': event_counts.get('TakeOn', 0),
            'tackles': event_counts.get('Tackle', 0),
            'interceptions': event_counts.get('Interception', 0),
            'ball_recoveries': event_counts.get('BallRecovery', 0),
            'fouls': event_counts.get('Foul', 0),
            # Stats from players.csv
            'aerials_won': info.get('aerials_won', 0),
            'aerials_total': info.get('aerials_total', 0),
            'shots_total': info.get('shots_total', 0),
            'shots_on_target': info.get('shots_on_target', 0),
            'corners_total': info.get('corners_total', 0),
            'dribbles_won': info.get('dribbles_won', 0),
            'clearances': info.get('clearances', 0),
            'tackles': info.get('tackles', 0),
            'saves': info.get('saves', 0),
            'touches': info.get('touches', 0),
            'dispossessed': info.get('dispossessed', 0),
        })
    
    return pd.DataFrame(node_rows)


def count_cards(events):
    yellow = 0
    red = 0
    if 'cardType' in events.columns:
        for card_type in events['cardType']:
            if card_type is None or pd.isna(card_type):
                continue
            if isinstance(card_type, dict):
                card_name = card_type.get('displayName', '').lower()
                if 'yellow' in card_name:
                    yellow += 1
                elif 'red' in card_name:
                    red += 1
    return yellow, red


INGAME_EVENT_MAP = {
    "ball_touches": ["BallTouch"],
    "substitutions": ["SubstitutionOn"],
    "aerials": ["Aerial"],
    "ball_recoveries": ["BallRecovery"],
    "blocked_passes": ["BlockedPass"],
    "clearances": ["Clearance"],
    "corners_awarded": ["CornerAwarded"],
    "dispossessions": ["Dispossessed"],
    "fouls": ["Foul"],
    "interceptions": ["Interception"],
    "missed_shots": ["MissedShots"],
    "saved_shots": ["SavedShot"],
    "tackles": ["Tackle"],
    "take_ons": ["TakeOn"],
    "offsides_provoked": ["OffsideProvoked"],
}


def count_event_types(events_df, event_names):
    if events_df is None or events_df.empty or 'type_name' not in events_df.columns:
        return 0
    if isinstance(event_names, str):
        event_names = [event_names]
    mask = events_df['type_name'].isin(event_names)
    return int(mask.sum())


def collect_team_event_indicators(events_df):
    """
    Build the 16 team-level indicators described in the paper (touches, subs, aerials, etc.).
    Returns a dict keyed by indicator name along with yellow/red splits for compatibility.
    """
    indicators = {name: count_event_types(events_df, names) for name, names in INGAME_EVENT_MAP.items()}
    yellow, red = count_cards(events_df if events_df is not None else pd.DataFrame())
    indicators["cards"] = yellow + red
    return indicators, yellow, red

def is_shot_in_box(shot_event):
    if 'x' not in shot_event or 'y' not in shot_event:
        return False
    x = shot_event['x']
    y = shot_event['y']
    if pd.isna(x) or pd.isna(y):
        return False
    try:
        x_val = float(x)
        y_val = float(y)
    except (ValueError, TypeError):
        return False
    return x_val > 83 and 25 < y_val < 75

    # NOT: summarize_positions ve append_player_positions fonksiyonları kaldırıldı
    # çünkü WhoScored JSON'unda avgX, avgY (oyuncu pozisyon koordinatları) mevcut değil

successful = 0
failed = 0
processed_match_ids = set()


for idx, match_info in enumerate(all_matches, 1):
    match_id = match_info['match_id']
    if match_id in processed_match_ids:
        print(f"\n[{idx}/{len(all_matches)}] Match {match_id}: ⏭️  Skipped (already processed)")
        continue

    league = match_info.get('league', 'Unknown')

    # SADECE Championship maçlarını işle
    if league != "ENG-Championship":
        print(f"\n[{idx}/{len(all_matches)}] Match {match_id}: ⏭️  Skipped (league {league})")
        continue

    # Lig klasörü adını belirle
    league_dir_name = "championship"

    home_team = match_info.get('home_team', '')
    away_team = match_info.get('away_team', '')
    print(f"\n[{idx}/{len(all_matches)}] Match {match_id}: {home_team} vs {away_team} ({league})")
    
    # Kontrol mantığı: Eğer 45, 60, 75, 90 varsa sadece 30'u çek
    # SADECE championship klasörüne bak (bu dosya sadece Championship için)
    def has_all_required_files(folder: Path) -> bool:
        return folder.exists() and all((folder / name).exists() for name in REQUIRED_FILENAMES)
    
    # Önce 45, 60, 75, 90'ın hepsinin olup olmadığını kontrol et (sadece championship klasöründe)
    existing_periods_all_present = True
    for time_cutoff in EXISTING_PERIODS_TO_CHECK:
        time_output_dir = base_data_dir / f"minute_{time_cutoff}" / league_dir_name
        match_dir = time_output_dir / f"match_{match_id}"
        if not has_all_required_files(match_dir):
            existing_periods_all_present = False
            break
    
    # Eğer 45, 60, 75, 90 varsa, sadece 30'u kontrol et (sadece championship klasöründe)
    if existing_periods_all_present:
        time_30_dir = base_data_dir / f"minute_30" / league_dir_name
        match_30_dir = time_30_dir / f"match_{match_id}"
        if has_all_required_files(match_30_dir):
            print(f"   ⏭️  All time periods (including 30min) already processed, skipping match")
            processed_match_ids.add(match_id)
            successful += 1
            continue
        # 30 dakikalık veriler yoksa, sadece 30'u işle
        print(f"   ℹ️  45, 60, 75, 90 min data exists. Processing only 30 min data...")
        # TIME_PERIODS'u geçici olarak sadece 30 yap
        periods_to_process = [30]
    else:
        # Normal işleme: Eksik olan tüm zaman dilimlerini işle
        periods_to_process = TIME_PERIODS
    
    # Match data'yı bir kez çek (tüm zaman dilimleri için ortak)
    reader = get_match_data_with_retry(match_id, max_retries=3)
    if reader is None:
        print("   ❌ Failed to retrieve data after retries")
        failed += 1
        continue
    reader.seek(0)
    json_data = json.load(reader)
    if json_data is None or 'events' not in json_data:
        print("   ⚠️  No data retrieved")
        failed += 1
        time.sleep(1)
        continue
    
    # Zaman dilimleri için işleme yap (periods_to_process: ya [30] ya da [30, 45, 60, 75, 90])
    for time_cutoff in periods_to_process:
        # Her zaman dilimi için ayrı klasör
        time_output_dir = base_data_dir / f"minute_{time_cutoff}" / league_dir_name
        time_output_dir.mkdir(parents=True, exist_ok=True)
        
        match_dir = time_output_dir / f"match_{match_id}"
        
        # Bu zaman dilimi için zaten tüm dosyalar var mı kontrol et
        if has_all_required_files(match_dir):
            print(f"   [{time_cutoff}min] ⏭️  Already processed")
            continue
        
        print(f"   [{time_cutoff}min] Processing...")
        
        match_dir.mkdir(exist_ok=True)
        
        passes_file = match_dir / "passes.csv"
        features_file = match_dir / "features.csv"
        events_file = match_dir / "events.csv"
        players_file = match_dir / "players.csv"
        player_positions_file = match_dir / "player_positions.csv"
        node_features_file = match_dir / "node_features.csv"
        event_counts_file = match_dir / "player_event_counts.csv"
        event_summary_file = match_dir / "player_event_summary.csv"
        
        try:
            # json_data zaten maç başında çekildi, sadece event'leri filtrele
            events = json_data['events']
            df_events_all = pd.DataFrame(events)
            
            # Zaman dilimine göre event'leri filtrele
            time_col = 'expandedMinute' if 'expandedMinute' in df_events_all.columns else 'minute' if 'minute' in df_events_all.columns else None
            if time_col:
                df_events = df_events_all[df_events_all[time_col] <= time_cutoff].copy()
            else:
                df_events = df_events_all.copy()
            
            if 'type' in df_events.columns:
                df_events['type_name'] = df_events['type'].apply(lambda x: x.get('displayName', '') if isinstance(x, dict) else str(x))
            player_dict = {}
            if 'playerIdNameDictionary' in json_data:
                player_dict = {int(k): v for k, v in json_data['playerIdNameDictionary'].items()}
            if 'playerId' in df_events.columns:
                df_events['playerId'] = df_events['playerId'].apply(safe_int)
                if 'player_name' not in df_events.columns:
                    df_events['player_name'] = None
                if player_dict:
                    df_events['player_name'] = df_events['player_name'].fillna(df_events['playerId'].map(player_dict))
            else:
                df_events['playerId'] = pd.Series(index=df_events.index, dtype='Int64')
                df_events['player_name'] = df_events.get('player_name', None)
            if 'relatedPlayerId' in df_events.columns:
                df_events['relatedPlayerId'] = df_events['relatedPlayerId'].apply(safe_int)
                if player_dict:
                    df_events['related_player_name'] = df_events['relatedPlayerId'].map(player_dict)
            else:
                df_events['relatedPlayerId'] = pd.Series(index=df_events.index, dtype='Int64')
            team_dict = {}
            if 'home' in json_data and 'away' in json_data:
                team_dict = {
                    int(json_data['home']['teamId']): json_data['home']['name'],
                    int(json_data['away']['teamId']): json_data['away']['name'],
                }
                if 'teamId' in df_events.columns:
                    df_events['teamId'] = df_events['teamId'].apply(safe_int)
                    df_events['team_name'] = df_events['teamId'].map(team_dict)
            if 'team_name' not in df_events.columns:
                df_events['team_name'] = None
            df_events['match_id'] = match_id
        
            # outcomeType, period ve type kolonlarını temizle (dictionary'den displayName'i al)
            # Bu işlemi add_receiver_columns'dan önce yapıyoruz çünkü is_successful fonksiyonu outcomeType'ı kullanıyor
            if 'outcomeType' in df_events.columns:
                df_events['outcomeType'] = df_events['outcomeType'].apply(get_display_value)
            if 'period' in df_events.columns:
                df_events['period'] = df_events['period'].apply(get_display_value)
            if 'type' in df_events.columns:
                # type zaten type_name'e çevrildi, ama orijinal type kolonunu da temizle
                df_events['type'] = df_events['type'].apply(get_display_value)
        
            df_events = add_receiver_columns(df_events, player_dict)
            if player_dict:
                df_events['player_name'] = df_events['player_name'].fillna(df_events['playerId'].map(player_dict))
                df_events['receiver_name'] = df_events['receiver_name'].fillna(df_events['receiver_id'].map(player_dict))
            if team_dict:
                df_events['team_name'] = df_events['team_name'].fillna(df_events['teamId'].map(team_dict))
            if 'qualifiers' in df_events.columns:
                df_events['qualifiers'] = df_events['qualifiers'].apply(
                    lambda val: json.dumps(val) if isinstance(val, (dict, list)) else val
                )
            if 'satisfiedEventsTypes' in df_events.columns:
                df_events['satisfiedEventsTypes'] = df_events['satisfiedEventsTypes'].apply(
                    lambda val: json.dumps(val) if isinstance(val, (dict, list)) else val
                )
            cols = list(df_events.columns)
            if 'team_name' in cols:
                cols.remove('team_name')
                cols.insert(1, 'team_name')
            if 'playerId' in cols and 'player_name' in cols:
                cols.remove('player_name')
                player_id_idx = cols.index('playerId')
                cols.insert(player_id_idx + 1, 'player_name')
            receiver_insert_idx = None
            if 'receiver_id' in cols and 'receiver_name' in cols:
                cols.remove('receiver_name')
                receiver_id_idx = cols.index('receiver_id')
                cols.insert(receiver_id_idx + 1, 'receiver_name')
                receiver_insert_idx = cols.index('receiver_name')
            if 'possession_retained' in cols:
                cols.remove('possession_retained')
                insert_idx = (receiver_insert_idx + 1) if receiver_insert_idx is not None else len(cols)
                cols.insert(insert_idx, 'possession_retained')
            df_events = df_events[cols]
            df_events.to_csv(events_file, index=False)
            passes = df_events[df_events['type_name'] == 'Pass'].copy()
            if 'minute' in passes.columns and 'second' in passes.columns:
                cols = list(passes.columns)
                if 'minute' in cols:
                    cols.remove('minute')
                if 'second' in cols:
                    cols.remove('second')
                cols.insert(0, 'second')
                cols.insert(0, 'minute')
                passes = passes[cols]
            passes.to_csv(passes_file, index=False)
            player_rows_info = []
            home_players = extract_player_rows(match_id, 'home', json_data.get('home'))
            away_players = extract_player_rows(match_id, 'away', json_data.get('away'))
            player_rows_info.extend(home_players)
            player_rows_info.extend(away_players)
        
            # WhoScored'da rating bazen ayrı bir yerde olabilir - playerRatings dictionary'sinde kontrol et
            if 'playerRatings' in json_data and isinstance(json_data['playerRatings'], dict):
                ratings_dict = json_data['playerRatings']
                for player_row in player_rows_info:
                    player_id = player_row.get('player_id')
                    if player_id and player_id in ratings_dict:
                        rating_value = ratings_dict[player_id]
                        if isinstance(rating_value, dict):
                            rating_value = rating_value.get('value') or rating_value.get('rating')
                        if player_row.get('rating') is None or pd.isna(player_row.get('rating')):
                            player_row['rating'] = safe_float(rating_value)
        
            # Rating verilerini JSON'dan daha detaylı ara
            # WhoScored'da rating bazen farklı yerlerde olabilir
            if len(player_rows_info) > 0:
                ratings_found = sum(1 for p in player_rows_info if p.get('rating') is not None and pd.notna(p.get('rating')))
            
            # Eğer rating bulunamadıysa, JSON'daki diğer yerleri kontrol et
            if ratings_found == 0:
                # playerStats veya playerRatings gibi ayrı bir dictionary olabilir
                if 'playerStats' in json_data:
                    player_stats = json_data['playerStats']
                    if isinstance(player_stats, dict):
                        for player_row in player_rows_info:
                            player_id = player_row.get('player_id')
                            if player_id and str(player_id) in player_stats:
                                stat_data = player_stats[str(player_id)]
                                if isinstance(stat_data, dict) and 'rating' in stat_data:
                                    rating_val = stat_data['rating']
                                    if isinstance(rating_val, dict):
                                        rating_val = rating_val.get('value')
                                    player_row['rating'] = safe_float(rating_val)
                
                # home/away içindeki players array'inde rating olabilir
                for team_key in ['home', 'away']:
                    if team_key in json_data and 'players' in json_data[team_key]:
                        for player_json in json_data[team_key]['players']:
                            player_json_id = safe_int(player_json.get('playerId'))
                            if player_json_id:
                                # JSON'daki player'ı bul
                                for player_row in player_rows_info:
                                    if player_row.get('player_id') == player_json_id:
                                        # Rating'i tekrar dene, bu sefer daha detaylı
                                        rating = extract_player_rating(player_json, player_json.get('stats', {}))
                                        if rating is not None and pd.notna(rating):
                                            player_row['rating'] = rating
                                        break
                
                # Tekrar kontrol et
                ratings_found = sum(1 for p in player_rows_info if p.get('rating') is not None and pd.notna(p.get('rating')))
        
            # Event'lerden gol ve asist sayılarını hesapla
            # Goal event'leri: type_name == 'Goal'
            # Assist: qualifiers içinde 'IntentionalGoalAssist' veya 'Assist'
            player_goals = {}
            player_assists = {}
        
            for _, event in df_events.iterrows():
                player_id = safe_int(event.get('playerId'))
                if player_id is None:
                    continue
                
                event_type = event.get('type_name', '')
                
                # Goal sayısı
                if event_type == 'Goal':
                    player_goals[player_id] = player_goals.get(player_id, 0) + 1
                
                # Assist sayısı (qualifiers'dan kontrol et)
                qualifiers = event.get('qualifiers', '')
                if qualifiers:
                    qual_str = str(qualifiers)
                    if 'IntentionalGoalAssist' in qual_str:
                        player_assists[player_id] = player_assists.get(player_id, 0) + 1
        
            # Player rows'a gol ve asist ekle
            for player_row in player_rows_info:
                pid = player_row.get('player_id')
                player_row['goals'] = player_goals.get(pid, 0)
                player_row['assists'] = player_assists.get(pid, 0)
            
            # Player rating'lerini time_cutoff'a göre güncelle
            # WhoScored'da player rating'leri dakika bazlı olarak stats.ratings içinde saklanıyor
            for team_key in ['home', 'away']:
                if team_key in json_data and 'players' in json_data[team_key]:
                    for player_json in json_data[team_key]['players']:
                        player_json_id = safe_int(player_json.get('playerId'))
                        if player_json_id:
                            # Player row'u bul
                            for player_row in player_rows_info:
                                if player_row.get('player_id') == player_json_id:
                                    # stats.ratings dictionary'sinden time_cutoff'a kadar olan rating'i al
                                    stats = player_json.get('stats', {}) or {}
                                    ratings = stats.get('ratings')
                                    if ratings and isinstance(ratings, dict):
                                        # time_cutoff'a kadar olan rating'leri filtrele
                                        rating_values = []
                                        for key, val in ratings.items():
                                            if isinstance(val, (int, float)):
                                                minute = int(key) if str(key).isdigit() else 0
                                                if minute <= time_cutoff:
                                                    rating_values.append((minute, float(val)))
                                        
                                        if rating_values:
                                            # time_cutoff'a en yakın rating'i al
                                            rating_values.sort(key=lambda x: x[0])
                                            time_specific_rating = rating_values[-1][1]
                                            player_row['rating'] = safe_float(time_specific_rating)
                                    break
        
            player_info_df = pd.DataFrame(player_rows_info)
            if player_info_df.empty:
                player_info_df = pd.DataFrame(
                columns=[
                    'match_id',
                    'team_side',
                    'team_id',
                    'team_name',
                    'player_id',
                    'player_name',
                    'shirt_no',
                    'position',
                    'height_cm',
                    'weight_kg',
                    'age',
                    'is_first_eleven',
                    'is_man_of_the_match',
                    'rating',
                ]
            )
            player_info_df.to_csv(players_file, index=False)
            player_event_counts_df, player_event_summary_df, player_event_counter = build_player_event_tables(match_id, df_events)
            player_event_counts_df.to_csv(event_counts_file, index=False)
            player_event_summary_df.to_csv(event_summary_file, index=False)
            features = {}
            features['match_id'] = match_id
            features['league'] = league
            if 'home' in json_data and 'away' in json_data:
                features['home_team'] = json_data['home']['name']
                features['away_team'] = json_data['away']['name']
                home_team_id = int(json_data['home']['teamId'])
                away_team_id = int(json_data['away']['teamId'])
                features['home_team_id'] = home_team_id
                features['away_team_id'] = away_team_id
                # Takım bazında rating'i stats.ratings (dakika bazlı dict) içinden time_cutoff'a kadar olan rating'i al
                home_rating = None
                away_rating = None
                
                # stats içinde ratings (dakika bazlı dict) - time_cutoff'a kadar olan rating'i al
                home_stats = json_data['home'].get('stats', {})
                if home_stats and 'ratings' in home_stats:
                    ratings_dict = home_stats['ratings']
                    if isinstance(ratings_dict, dict):
                        # time_cutoff'a kadar olan rating'leri filtrele
                        rating_values = []
                        for key, val in ratings_dict.items():
                            if isinstance(val, (int, float)):
                                minute = int(key) if str(key).isdigit() else 0
                                if minute <= time_cutoff:
                                    rating_values.append((minute, float(val)))
                        if rating_values:
                            rating_values.sort(key=lambda x: x[0])
                            # time_cutoff'a en yakın rating'i al
                            home_rating = rating_values[-1][1] if rating_values else None
                
                away_stats = json_data['away'].get('stats', {})
                if away_stats and 'ratings' in away_stats:
                    ratings_dict = away_stats['ratings']
                    if isinstance(ratings_dict, dict):
                        # time_cutoff'a kadar olan rating'leri filtrele
                        rating_values = []
                        for key, val in ratings_dict.items():
                            if isinstance(val, (int, float)):
                                minute = int(key) if str(key).isdigit() else 0
                                if minute <= time_cutoff:
                                    rating_values.append((minute, float(val)))
                        if rating_values:
                            rating_values.sort(key=lambda x: x[0])
                            # time_cutoff'a en yakın rating'i al
                            away_rating = rating_values[-1][1] if rating_values else None
                
                features['home_team_rating'] = safe_float(home_rating) if home_rating is not None else 0.0
                features['away_team_rating'] = safe_float(away_rating) if away_rating is not None else 0.0
            else:
                home_team_id = None
                away_team_id = None
                features['home_team_rating'] = 0.0
                features['away_team_rating'] = 0.0
            ft_score = json_data.get('ftScore')
            if ft_score:
                score_parts = str(ft_score).split(':')
                if len(score_parts) == 2:
                    features['home_score'] = int(score_parts[0].strip())
                    features['away_score'] = int(score_parts[1].strip())
                else:
                    features['home_score'] = 0
                    features['away_score'] = 0
            else:
                features['home_score'] = 0
                features['away_score'] = 0
            if features['home_score'] > features['away_score']:
                features['result'] = 'home_win'
            elif features['away_score'] > features['home_score']:
                features['result'] = 'away_win'
            else:
                features['result'] = 'draw'
            home_events = df_events[df_events['teamId'] == home_team_id] if home_team_id is not None else pd.DataFrame()
            away_events = df_events[df_events['teamId'] == away_team_id] if away_team_id is not None else pd.DataFrame()

            # collect_team_event_indicators zaten INGAME_EVENT_MAP'i kullanarak ball_touches'u 
            # BallTouch event'lerinden sayıyor (PDF'teki gibi)
            home_indicators, home_yellow, home_red = collect_team_event_indicators(home_events)
            away_indicators, away_yellow, away_red = collect_team_event_indicators(away_events)
            for key, value in home_indicators.items():
                features[f'home_{key}'] = value
            for key, value in away_indicators.items():
                features[f'away_{key}'] = value
            
            # Aerials: Event'lerden hesapla (time_cutoff'a göre)
            # aerials_total = Tüm Aerial event'lerinin sayısı
            # aerials_won = outcomeType == "Successful" olan Aerial event'lerinin sayısı
            home_aerial_events = home_events[home_events['type_name'] == 'Aerial'] if not home_events.empty else pd.DataFrame()
            away_aerial_events = away_events[away_events['type_name'] == 'Aerial'] if not away_events.empty else pd.DataFrame()
            
            features['home_aerials_total'] = len(home_aerial_events)
            features['away_aerials_total'] = len(away_aerial_events)
            
            # Aerials won: Successful outcomeType'a sahip Aerial event'leri
            if not home_aerial_events.empty and 'outcomeType' in home_aerial_events.columns:
                home_aerials_won = len(home_aerial_events[home_aerial_events['outcomeType'] == 'Successful'])
            else:
                home_aerials_won = 0
            features['home_aerials_won'] = home_aerials_won
            
            if not away_aerial_events.empty and 'outcomeType' in away_aerial_events.columns:
                away_aerials_won = len(away_aerial_events[away_aerial_events['outcomeType'] == 'Successful'])
            else:
                away_aerials_won = 0
            features['away_aerials_won'] = away_aerials_won
            home_passes = home_events[home_events['type_name'] == 'Pass'] if not home_events.empty else pd.DataFrame()
            away_passes = away_events[away_events['type_name'] == 'Pass'] if not away_events.empty else pd.DataFrame()
            features['home_total_passes'] = len(home_passes)
            features['away_total_passes'] = len(away_passes)
            home_success = home_passes[home_passes['outcomeType'].apply(is_successful)] if 'outcomeType' in home_passes.columns else pd.DataFrame()
            away_success = away_passes[away_passes['outcomeType'].apply(is_successful)] if 'outcomeType' in away_passes.columns else pd.DataFrame()
            features['home_successful_passes'] = len(home_success)
            features['away_successful_passes'] = len(away_success)
            features['home_pass_accuracy'] = len(home_success) / len(home_passes) if len(home_passes) else 0
            features['away_pass_accuracy'] = len(away_success) / len(away_passes) if len(away_passes) else 0
            shot_types = ['MissedShots', 'SavedShot', 'Goal', 'ShotOnPost']
            home_shots = home_events[home_events['type_name'].isin(shot_types)] if not home_events.empty else pd.DataFrame()
            away_shots = away_events[away_events['type_name'].isin(shot_types)] if not away_events.empty else pd.DataFrame()
            features['home_shots'] = len(home_shots)
            features['away_shots'] = len(away_shots)
            on_target_types = ['SavedShot', 'Goal', 'ShotOnPost']
            home_sot = home_events[home_events['type_name'].isin(on_target_types)] if not home_events.empty else pd.DataFrame()
            away_sot = away_events[away_events['type_name'].isin(on_target_types)] if not away_events.empty else pd.DataFrame()
            features['home_shots_on_target'] = len(home_sot)
            features['away_shots_on_target'] = len(away_sot)
            home_dribbles = home_events[home_events['type_name'] == 'TakeOn'] if not home_events.empty else pd.DataFrame()
            away_dribbles = away_events[away_events['type_name'] == 'TakeOn'] if not away_events.empty else pd.DataFrame()
            features['home_dribbles'] = len(home_dribbles)
            features['away_dribbles'] = len(away_dribbles)
            def split_thirds(dribbles, prefix):
                if dribbles.empty or 'x' not in dribbles.columns:
                    features[f'{prefix}_dribbles_attacking_third'] = 0
                    features[f'{prefix}_dribbles_middle_third'] = 0
                    features[f'{prefix}_dribbles_defensive_third'] = 0
                    return
                dribble_locs = dribbles[['x', 'y']].dropna()
                features[f'{prefix}_dribbles_attacking_third'] = len(dribble_locs[dribble_locs['x'] > 66.67])
                features[f'{prefix}_dribbles_middle_third'] = len(dribble_locs[(dribble_locs['x'] > 33.33) & (dribble_locs['x'] <= 66.67)])
                features[f'{prefix}_dribbles_defensive_third'] = len(dribble_locs[dribble_locs['x'] <= 33.33])
            split_thirds(home_dribbles, 'home')
            split_thirds(away_dribbles, 'away')
            home_recoveries = home_events[home_events['type_name'] == 'BallRecovery'] if not home_events.empty else pd.DataFrame()
            away_recoveries = away_events[away_events['type_name'] == 'BallRecovery'] if not away_events.empty else pd.DataFrame()
            features['home_ball_recoveries'] = len(home_recoveries)
            features['away_ball_recoveries'] = len(away_recoveries)
            home_tackles = home_events[home_events['type_name'] == 'Tackle'] if not home_events.empty else pd.DataFrame()
            away_tackles = away_events[away_events['type_name'] == 'Tackle'] if not away_events.empty else pd.DataFrame()
            features['home_tackles'] = len(home_tackles)
            features['away_tackles'] = len(away_tackles)
            home_interceptions = home_events[home_events['type_name'] == 'Interception'] if not home_events.empty else pd.DataFrame()
            away_interceptions = away_events[away_events['type_name'] == 'Interception'] if not away_events.empty else pd.DataFrame()
            features['home_interceptions'] = len(home_interceptions)
            features['away_interceptions'] = len(away_interceptions)
            # Kornerleri event'lerden saymayı deneyelim, ama player stats'tan da alacağız
            home_corners = home_events[home_events['type_name'] == 'Corner'] if not home_events.empty else pd.DataFrame()
            away_corners = away_events[away_events['type_name'] == 'Corner'] if not away_events.empty else pd.DataFrame()
            # Önce event'lerden sayıyoruz, sonra player stats'tan gelen değerlerle override edeceğiz
            features['home_corners'] = len(home_corners)
            features['away_corners'] = len(away_corners)
            features['home_yellow_cards'] = home_yellow
            features['away_yellow_cards'] = away_yellow
            features['home_red_cards'] = home_red
            features['away_red_cards'] = away_red
            # Offsides - OffsideGiven event type
            home_offsides = home_events[home_events['type_name'].isin(['Offside', 'OffsideGiven', 'OffsidePass'])] if not home_events.empty else pd.DataFrame()
            away_offsides = away_events[away_events['type_name'].isin(['Offside', 'OffsideGiven', 'OffsidePass'])] if not away_events.empty else pd.DataFrame()
            features['home_offsides'] = len(home_offsides)
            features['away_offsides'] = len(away_offsides)
            home_shots_in_box = home_shots[home_shots.apply(is_shot_in_box, axis=1)] if not home_shots.empty else pd.DataFrame()
            away_shots_in_box = away_shots[away_shots.apply(is_shot_in_box, axis=1)] if not away_shots.empty else pd.DataFrame()
            features['home_shots_in_box'] = len(home_shots_in_box)
            features['away_shots_in_box'] = len(away_shots_in_box)
            # NOT: xG (Expected Goals) WhoScored JSON'unda mevcut değil - kaldırıldı
            total_passes = features['home_total_passes'] + features['away_total_passes']
            if total_passes > 0:
                features['home_possession_pct'] = (features['home_total_passes'] / total_passes) * 100
                features['away_possession_pct'] = (features['away_total_passes'] / total_passes) * 100
            else:
                features['home_possession_pct'] = 50.0
                features['away_possession_pct'] = 50.0
            def get_starting_formation(team_formations):
                if not team_formations:
                    return "Unknown"
                for formation in team_formations:
                        if formation.get('startMinuteExpanded', -1) == 0:
                            return formation.get('formationName', 'Unknown')
                return team_formations[0].get('formationName', 'Unknown')
            home_formations = json_data.get('home', {}).get('formations', [])
            away_formations = json_data.get('away', {}).get('formations', [])
            features['home_formation'] = get_starting_formation(home_formations)
            features['away_formation'] = get_starting_formation(away_formations)
        
            # NOT: WhoScored JSON'unda avgX, avgY (oyuncu pozisyon koordinatları) mevcut değil
            # Bu yüzden formation_count, formation_avg_x, formation_avg_y hesaplanamıyor
        
            # Player stats'tan toplam değerleri hesapla (aerials, clearances, saves, corners, dispossessed)
            # NOT: blocks, key_passes, assists, goals WhoScored JSON'unda mevcut değil - kaldırıldı
            def sum_player_stats(team_key):
                total_stats = {
                    'aerials_won': 0,
                    'aerials_total': 0,
                    'clearances': 0,
                    'saves': 0,
                    'corners_total': 0,
                    'corners_accurate': 0,
                    'dispossessed': 0,
                    'touches': 0,
                }
                if team_key in json_data and 'players' in json_data[team_key]:
                    for player in json_data[team_key]['players']:
                        stats = player.get('stats', {}) or {}
                        player_stats = extract_player_stats_from_dict(stats)
                        total_stats['aerials_won'] += player_stats.get('aerialsWon', 0)
                        total_stats['aerials_total'] += player_stats.get('aerialsTotal', 0)
                        total_stats['clearances'] += player_stats.get('clearances', 0)
                        total_stats['saves'] += player_stats.get('saves', 0) or player_stats.get('totalSaves', 0)
                        total_stats['corners_total'] += player_stats.get('cornersTotal', 0) or player_stats.get('corners', 0)
                        total_stats['corners_accurate'] += player_stats.get('cornersAccurate', 0)
                        total_stats['dispossessed'] += player_stats.get('dispossessed', 0)
                        # WhoScored team heatmaps \"Touches\" sayısı: oyuncu bazlı touches toplamı
                        total_stats['touches'] += player_stats.get('touches', 0)
                return total_stats
        
            home_stats = sum_player_stats('home')
            away_stats = sum_player_stats('away')
        
            # Aerials artık event'lerden hesaplanıyor (time_cutoff'a göre)
            # Bu yüzden player stats'tan almayı kaldırdık
            # features['home_aerials_won'], features['away_aerials_won'],
            # features['home_aerials_total'], features['away_aerials_total'] 
            # yukarıda event'lerden set edildi
            features['home_clearances'] = home_stats['clearances']
            features['away_clearances'] = away_stats['clearances']
            features['home_saves'] = home_stats['saves']
            features['away_saves'] = away_stats['saves']
            # Ball touches artık event sayısından hesaplanıyor (time_cutoff'a göre)
            # Bu yüzden burada player stats'tan almayı kaldırdık
            # features['home_ball_touches'] ve features['away_ball_touches'] yukarıda event sayısından set edildi
            # Kornerleri player stats'tan al (event'lerden sayılan değer 0 ise veya player stats'tan gelen değer daha büyükse)
            if home_stats['corners_total'] > 0:
                features['home_corners'] = home_stats['corners_total']
            if away_stats['corners_total'] > 0:
                features['away_corners'] = away_stats['corners_total']
            features['home_corners_accurate'] = home_stats['corners_accurate']
            features['away_corners_accurate'] = away_stats['corners_accurate']
            features['home_dispossessed'] = home_stats['dispossessed']
            features['away_dispossessed'] = away_stats['dispossessed']
        
            # Player positions - Event'lerden average X, Y hesapla (PDF'teki yöntem)
            # Sadece starting eleven oyuncuları için
            home_starting_ids = get_starting_eleven_ids(json_data.get('home', {}))
            away_starting_ids = get_starting_eleven_ids(json_data.get('away', {}))
        
            home_positions = calculate_player_positions(df_events, home_team_id, home_starting_ids, time_cutoff=time_cutoff)
            away_positions = calculate_player_positions(df_events, away_team_id, away_starting_ids, time_cutoff=time_cutoff)
        
            # Formation avg x, y ekle
            features['home_starting_eleven_count'] = len(home_positions)
            features['away_starting_eleven_count'] = len(away_positions)
            features['home_avg_x'] = np.mean([p['avg_x'] for p in home_positions]) if home_positions else 0
            features['home_avg_y'] = np.mean([p['avg_y'] for p in home_positions]) if home_positions else 0
            features['away_avg_x'] = np.mean([p['avg_x'] for p in away_positions]) if away_positions else 0
            features['away_avg_y'] = np.mean([p['avg_y'] for p in away_positions]) if away_positions else 0
        
            # Player positions CSV
            position_rows = []
            position_rows.extend(append_player_positions(match_id, 'home', features.get('home_team', ''), home_team_id, home_positions))
            position_rows.extend(append_player_positions(match_id, 'away', features.get('away_team', ''), away_team_id, away_positions))
            player_positions_df = pd.DataFrame(position_rows)
            if player_positions_df.empty:
                player_positions_df = pd.DataFrame(
                    columns=['match_id', 'team', 'team_name', 'team_id', 'player_id', 'player_name', 'avg_x', 'avg_y']
                )
            player_positions_df.to_csv(player_positions_file, index=False)
        
            # Node features CSV (PDF Table 3)
            node_features_df = build_node_features(
                match_id,
                player_positions_df,
                passes,
                player_info_df,
                player_event_counter,
            )
            if node_features_df.empty:
                node_features_df = pd.DataFrame(
                    columns=[
                        'match_id', 'team', 'team_id', 'team_name', 'player_id', 'player_name',
                        'position_encoded', 'position', 'shirt_no', 'height_cm', 'weight_kg', 'is_first_eleven', 'is_captain',
                        'rating', 'avg_x', 'avg_y', 'passes_made', 'passes_completed', 'passes_received',
                        'pass_accuracy', 'ball_touches', 'take_ons', 'tackles', 'interceptions',
                        'ball_recoveries', 'fouls', 'shots'
                    ]
                )
            node_features_df.to_csv(node_features_file, index=False)
            
            features_df = pd.DataFrame([features])
            features_df.to_csv(match_dir / "features.csv", index=False)
            print(f"   [{time_cutoff}min] ✅ Saved: {len(passes)} passes, {len(df_events)} events, {len(player_info_df)} players")
            time.sleep(0.5)
        except Exception as e:
            print(f"   [{time_cutoff}min] ❌ Error: {e}")
            failed += 1
            time.sleep(2)
            continue  # Bu zaman dilimi için hata, diğer zaman dilimlerini dene
    
    # Bu maç için tüm zaman dilimleri işlendi
    processed_match_ids.add(match_id)
    successful += 1
    if idx < len(all_matches):
        time.sleep(1)

print("\n" + "=" * 60)
print("✅ Processing complete!")
print(f"   Successful: {successful}")
print(f"   Failed: {failed}")
print(f"   Data saved to: {base_data_dir}")
print("=" * 60)
