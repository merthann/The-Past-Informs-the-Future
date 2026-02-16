"""
Process Other Leagues Data for Multiple Time Periods - 22/23 Season

Processes 22/23 season matches from bundesliga, la_liga, ligue_1, serie_a.
Outputs to dataotherleagues/minute{X}/{league}/match_{id}/

Features:
- Time-filtered player ratings (e.g., for 45min, uses rating from 0-45 only)
- Skips matches already processed
- Outputs to dataotherleagues/minute{X}/{league}/match_{id}/

Usage:
    python process_other_leagues_2223.py
    python process_other_leagues_2223.py --time_periods 45,60,75
    python process_other_leagues_2223.py --leagues bundesliga,la_liga
"""

import sys
import json
import re
import time
import argparse
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
project_root = script_dir.parent

# Match ID dosyası - 22/23 sezon
match_ids_file = script_dir / "match_ids_other_leagues_2223.json"
if not match_ids_file.exists():
    print(f"❌ Match IDs file not found: {match_ids_file}")
    print(f"   Run: python collect_match_ids_other_leagues.py")
    sys.exit(1)

print(f"📁 Using match IDs file: {match_ids_file.name}")

with open(match_ids_file, 'r') as f:
    all_matches = json.load(f)

# Position label encoding
POSITION_ENCODING = {
    "GK": 0, "DC": 1, "DL": 2, "DR": 3, "DMC": 4, "DML": 5, "DMR": 6,
    "MC": 7, "ML": 8, "MR": 9, "AMC": 10, "AML": 11, "AMR": 12,
    "FW": 13, "FWL": 14, "FWR": 15,
}
UNKNOWN_POSITION_CODE = max(POSITION_ENCODING.values()) + 1

# League mapping
LEAGUE_MAP = {
    "GER-Bundesliga": "bundesliga",
    "ESP-La Liga": "la_liga",
    "FRA-Ligue 1": "ligue_1",
    "ITA-Serie A": "serie_a",
}

# Time periods to process
DEFAULT_TIME_PERIODS = [45, 60, 75, 90]

# Required CSV files per match
REQUIRED_FILENAMES = [
    "events.csv",
    "players.csv", 
    "features.csv",
    "player_positions.csv",
    "passes.csv",
]

# NEW: Output to dataotherleagues directory (not data/)
base_data_dir = project_root / "dataotherleagues"


import os

def create_scraper():
    # Chrome Beta yolunu sisteme dikte ediyoruz
    os.environ['UC_CHROME_BINARY'] = r"C:\Program Files\Google\Chrome Beta\Application\chrome.exe"
    return sd.WhoScored(headless=True)

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
    if not position or not isinstance(position, str):
        return UNKNOWN_POSITION_CODE
    pos = position.strip().upper()
    return POSITION_ENCODING.get(pos, UNKNOWN_POSITION_CODE)


def sanitize_event_column(name: str) -> str:
    label = str(name) if name is not None else "unknown"
    label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
    return f"event_{label}" if label else "event_unknown"


def extract_player_rating_at_time(player_data, stats=None, time_cutoff=90):
    """Extract player rating UP TO the specified time cutoff."""
    rating = None
    
    if rating is None and stats:
        ratings = stats.get("ratings")
        if ratings is not None:
            if isinstance(ratings, dict):
                rating_values = []
                for key, val in ratings.items():
                    if isinstance(val, (int, float)):
                        minute = int(key) if key.isdigit() else 0
                        if minute <= time_cutoff:
                            rating_values.append((minute, float(val)))
                
                if rating_values:
                    rating_values.sort(key=lambda x: x[0])
                    rating = rating_values[-1][1]
                else:
                    rating = ratings.get("value") or ratings.get("rating") or ratings.get("overall")
            elif isinstance(ratings, (int, float)):
                rating = ratings
    
    if rating is None:
        rating = player_data.get("rating")
        if isinstance(rating, dict):
            rating = rating.get("value") or rating.get("displayName")
    
    if rating is None and stats:
        rating_stat = stats.get("rating")
        if isinstance(rating_stat, dict):
            rating = rating_stat.get("value") or rating_stat.get("displayName")
        elif rating_stat is not None:
            rating = rating_stat
    
    if rating is None:
        ratings = player_data.get("ratings")
        if isinstance(ratings, dict):
            rating = ratings.get("value") or ratings.get("overall") or ratings.get("rating")
        elif isinstance(ratings, (int, float)):
            rating = ratings
    
    if rating is None:
        rating = player_data.get("performanceRating") or player_data.get("matchRating")
        if isinstance(rating, dict):
            rating = rating.get("value")
    
    if rating is None and stats:
        rating = stats.get("performanceRating") or stats.get("matchRating") or stats.get("overallRating")
        if isinstance(rating, dict):
            rating = rating.get("value")
    
    return safe_float(rating)


def extract_player_stats_from_dict(stats_dict, time_cutoff=90):
    """Extract player stats, filtering by time cutoff where applicable."""
    if not stats_dict or not isinstance(stats_dict, dict):
        return {}
    
    result = {}
    stat_keys = [
        'aerialsWon', 'aerialsTotal', 'shotsTotal', 'shotsOnTarget', 'shotsOffTarget', 'shotsBlocked',
        'passesAccurate', 'passesTotal', 'passSuccess', 'cornersTotal', 'cornersAccurate',
        'dribblesWon', 'dribblesAttempted', 'clearances', 'tacklesTotal', 'tackleSuccess', 'interceptions',
        'saves', 'totalSaves', 'touches', 'dispossessed', 'foulsCommited', 'possession',
    ]
    
    for key in stat_keys:
        value = stats_dict.get(key)
        if value is None:
            result[key] = 0
        elif isinstance(value, dict):
            total = 0
            for minute_key, minute_value in value.items():
                if isinstance(minute_value, (int, float)):
                    minute = int(minute_key) if minute_key.isdigit() else 0
                    if minute <= time_cutoff:
                        total += minute_value
            result[key] = total
        elif isinstance(value, (int, float)):
            result[key] = value
        else:
            result[key] = 0
    
    return result


def get_captain_id_from_formations(team_info):
    formations = team_info.get("formations", [])
    if formations:
        first_formation = formations[0]
        return safe_int(first_formation.get("captainPlayerId"))
    return None


def get_starting_eleven_ids(team_info):
    starting_ids = set()
    for player in team_info.get("players", []):
        if player.get("isFirstEleven", False):
            player_id = safe_int(player.get("playerId"))
            if player_id:
                starting_ids.add(player_id)
    return starting_ids


def extract_player_rows(match_id, team_side, team_info, time_cutoff=90):
    """Extract player data with time-filtered ratings."""
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
        rating = extract_player_rating_at_time(player, stats, time_cutoff)
        player_stats = extract_player_stats_from_dict(stats, time_cutoff)
        
        rows.append({
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
            "aerials_won": player_stats.get('aerialsWon', 0),
            "aerials_total": player_stats.get('aerialsTotal', 0),
            "shots_total": player_stats.get('shotsTotal', 0),
            "shots_on_target": player_stats.get('shotsOnTarget', 0),
            "shots_off_target": player_stats.get('shotsOffTarget', 0),
            "shots_blocked": player_stats.get('shotsBlocked', 0),
            "corners_total": player_stats.get('cornersTotal', 0),
            "corners_accurate": player_stats.get('cornersAccurate', 0),
            "dribbles_won": player_stats.get('dribblesWon', 0),
            "dribbles_attempted": player_stats.get('dribblesAttempted', 0),
            "clearances": player_stats.get('clearances', 0),
            "tackles": player_stats.get('tacklesTotal', 0),
            "tackle_success": player_stats.get('tackleSuccess', 0),
            "interceptions": player_stats.get('interceptions', 0),
            "saves": player_stats.get('saves', 0) or player_stats.get('totalSaves', 0),
            "touches": player_stats.get('touches', 0),
            "dispossessed": player_stats.get('dispossessed', 0),
            "fouls_commited": player_stats.get('foulsCommited', 0),
        })
    return rows


def get_match_data_with_retry(match_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            ws = create_scraper()
            match_url = f"https://www.whoscored.com/Matches/{match_id}/Live"
            filepath = ws.data_dir / "events" / f"direct_{match_id}.json"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            reader = ws.get(
                match_url, filepath,
                var="require.config.params['args'].matchCentreData",
                no_cache=False,
            )
            try:
                if hasattr(ws, '_driver') and ws._driver:
                    ws._driver.quit()
            except:
                pass
            return reader
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
            try:
                if 'ws' in locals() and hasattr(ws, '_driver') and ws._driver:
                    ws._driver.quit()
            except:
                pass
            if attempt < max_retries - 1:
                time.sleep(1)
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


def calculate_player_positions(df_events, team_id, time_cutoff=90):
    """Calculate avg positions for ALL players who have events."""
    if df_events.empty:
        return []
    
    time_col = 'expandedMinute' if 'expandedMinute' in df_events.columns else 'minute'
    if time_col in df_events.columns:
        df_events = df_events[df_events[time_col] <= time_cutoff].copy()
    
    team_events = df_events[df_events['teamId'] == team_id].copy()
    if team_events.empty:
        return []
    
    team_events = team_events.dropna(subset=['x', 'y'])
    team_events = team_events[team_events['x'] != 0]
    
    if team_events.empty:
        return []
    
    positions = []
    for player_id, group in team_events.groupby('playerId'):
        player_id_int = safe_int(player_id)
        if player_id_int is None:
            continue
        
        positions.append({
            'player_id': player_id_int,
            'player_name': group['player_name'].iloc[0] if 'player_name' in group.columns else '',
            'avg_x': float(group['x'].mean()),
            'avg_y': float(group['y'].mean()),
            'event_count': len(group),
        })
    
    return positions


def append_player_positions(match_id, team_label, team_name, team_id, players):
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


def count_cards(events):
    yellow, red = 0, 0
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
    "ball_touches": ["BallTouch"], "substitutions": ["SubstitutionOn"], "aerials": ["Aerial"],
    "ball_recoveries": ["BallRecovery"], "blocked_passes": ["BlockedPass"], "clearances": ["Clearance"],
    "corners_awarded": ["CornerAwarded"], "dispossessions": ["Dispossessed"], "fouls": ["Foul"],
    "interceptions": ["Interception"], "missed_shots": ["MissedShots"], "saved_shots": ["SavedShot"],
    "tackles": ["Tackle"], "take_ons": ["TakeOn"], "offsides_provoked": ["OffsideProvoked"],
}


def count_event_types(events_df, event_names):
    if events_df is None or events_df.empty or 'type_name' not in events_df.columns:
        return 0
    if isinstance(event_names, str):
        event_names = [event_names]
    return int(events_df['type_name'].isin(event_names).sum())


def collect_team_event_indicators(events_df):
    indicators = {name: count_event_types(events_df, names) for name, names in INGAME_EVENT_MAP.items()}
    yellow, red = count_cards(events_df if events_df is not None else pd.DataFrame())
    indicators["cards"] = yellow + red
    return indicators, yellow, red


def has_all_required_files(folder: Path) -> bool:
    return folder.exists() and all((folder / name).exists() for name in REQUIRED_FILENAMES)


def process_match_for_time(match_id, json_data, time_cutoff, output_dir, home_info, away_info):
    """Process a single match for a specific time cutoff."""
    match_dir = output_dir / f"match_{match_id}"
    
    if has_all_required_files(match_dir):
        return True  # Already processed
    
    match_dir.mkdir(parents=True, exist_ok=True)
    
    # Get events and filter by time
    events = json_data['events']
    df_events = pd.DataFrame(events)
    
    if 'type' in df_events.columns:
        df_events['type_name'] = df_events['type'].apply(
            lambda x: x.get('displayName', '') if isinstance(x, dict) else str(x)
        )
    
    # Time filter
    time_col = 'expandedMinute' if 'expandedMinute' in df_events.columns else 'minute'
    if time_col in df_events.columns:
        df_events = df_events[df_events[time_col] <= time_cutoff].copy()
    
    # Player dict
    player_dict = {}
    if 'playerIdNameDictionary' in json_data:
        player_dict = {int(k): v for k, v in json_data['playerIdNameDictionary'].items()}
    
    if 'playerId' in df_events.columns:
        df_events['playerId'] = df_events['playerId'].apply(safe_int)
        if 'player_name' not in df_events.columns:
            df_events['player_name'] = None
        if player_dict:
            df_events['player_name'] = df_events['player_name'].fillna(df_events['playerId'].map(player_dict))
    
    if 'relatedPlayerId' in df_events.columns:
        df_events['relatedPlayerId'] = df_events['relatedPlayerId'].apply(safe_int)
    
    # Add receiver columns
    df_events = add_receiver_columns(df_events, player_dict)
    
    # Extract players with time-filtered ratings
    home_players = extract_player_rows(match_id, "home", home_info, time_cutoff)
    away_players = extract_player_rows(match_id, "away", away_info, time_cutoff)
    all_players = home_players + away_players
    
    if not all_players:
        return False
    
    df_players = pd.DataFrame(all_players)
    
    # Get team IDs and names
    home_team_id = safe_int(home_info.get("teamId"))
    away_team_id = safe_int(away_info.get("teamId"))
    home_team_name = home_info.get("name", "Home")
    away_team_name = away_info.get("name", "Away")
    
    # Calculate player positions
    home_positions = calculate_player_positions(df_events, home_team_id, time_cutoff)
    away_positions = calculate_player_positions(df_events, away_team_id, time_cutoff)
    
    positions_rows = []
    positions_rows.extend(append_player_positions(match_id, "home", home_team_name, home_team_id, home_positions))
    positions_rows.extend(append_player_positions(match_id, "away", away_team_name, away_team_id, away_positions))
    df_positions = pd.DataFrame(positions_rows)
    
    # Extract passes
    df_passes = df_events[df_events['type_name'] == 'Pass'].copy()
    
    # Build features
    home_events = df_events[df_events['teamId'] == home_team_id]
    away_events = df_events[df_events['teamId'] == away_team_id]
    
    home_indicators, home_yellow, home_red = collect_team_event_indicators(home_events)
    away_indicators, away_yellow, away_red = collect_team_event_indicators(away_events)
    
    # Get result
    home_score, away_score = 0, 0
    score_str = json_data.get('score', '')
    if isinstance(score_str, str) and '-' in score_str:
        score_parts = score_str.split('-')
        if len(score_parts) >= 2:
            try:
                home_score = int(score_parts[0].strip())
                away_score = int(score_parts[1].strip())
            except (ValueError, TypeError):
                home_score, away_score = 0, 0
    
    if home_score > away_score:
        result = "home_win"
    elif away_score > home_score:
        result = "away_win"
    else:
        result = "draw"
    
    features_row = {
        'match_id': match_id,
        'home_team': home_team_name,
        'away_team': away_team_name,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_score': home_score,
        'away_score': away_score,
        'result': result,
        'time_cutoff': time_cutoff,
    }
    
    for key, value in home_indicators.items():
        features_row[f'home_{key}'] = value
    for key, value in away_indicators.items():
        features_row[f'away_{key}'] = value
    
    df_features = pd.DataFrame([features_row])
    
    # Save CSVs
    df_events.to_csv(match_dir / "events.csv", index=False)
    df_players.to_csv(match_dir / "players.csv", index=False)
    df_features.to_csv(match_dir / "features.csv", index=False)
    df_positions.to_csv(match_dir / "player_positions.csv", index=False)
    df_passes.to_csv(match_dir / "passes.csv", index=False)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Process Other Leagues Data - 22/23 Season")
    parser.add_argument("--time_periods", type=str, default="45,60,75,90",
                        help="Comma-separated time periods to process")
    parser.add_argument("--leagues", type=str, default="all",
                        help="Comma-separated leagues or 'all'")
    args = parser.parse_args()
    
    time_periods = [int(x) for x in args.time_periods.split(",")]
    
    if args.leagues.lower() == "all":
        leagues_to_process = list(LEAGUE_MAP.keys())
    else:
        leagues_to_process = [l.strip() for l in args.leagues.split(",")]
    
    print("=" * 60)
    print(f"Processing {len(all_matches)} Matches (22/23 Season)")
    print(f"Time periods: {time_periods}")
    print(f"Leagues: {leagues_to_process}")
    print(f"Output: {base_data_dir}")
    print("=" * 60)
    
    # Create output directories
    for time_period in time_periods:
        for league in LEAGUE_MAP.values():
            (base_data_dir / f"minute{time_period}" / league).mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    skipped = 0
    processed_match_ids = set()
    
    for idx, match_info in enumerate(all_matches, 1):
        match_id = match_info['match_id']
        if match_id in processed_match_ids:
            continue
        
        league = match_info.get('league', 'Unknown')
        
        if league not in LEAGUE_MAP:
            continue
        if league not in leagues_to_process and args.leagues.lower() != "all":
            continue
        
        league_dir_name = LEAGUE_MAP[league]
        home_team = match_info.get('home_team', '')
        away_team = match_info.get('away_team', '')
        
        # Check if already fully processed
        check_dir = base_data_dir / f"minute90" / league_dir_name / f"match_{match_id}"
        if has_all_required_files(check_dir):
            all_exist = True
            for time_cutoff in time_periods:
                if time_cutoff == 90:
                    continue
                period_dir = base_data_dir / f"minute{time_cutoff}" / league_dir_name / f"match_{match_id}"
                if not has_all_required_files(period_dir):
                    all_exist = False
                    break
            
            if all_exist:
                skipped += 1
                processed_match_ids.add(match_id)
                continue
        
        print(f"\n[{idx}/{len(all_matches)}] Match {match_id}: {home_team} vs {away_team} ({league})")
        
        # Get match data
        reader = get_match_data_with_retry(match_id, max_retries=3)
        if reader is None:
            print(f"   ❌ Failed to get match data")
            failed += 1
            processed_match_ids.add(match_id)
            continue
        
        # Parse JSON data (reader is a BytesIO object)
        try:
            reader.seek(0)
            json_data = json.load(reader)
            if isinstance(json_data, dict) and 'matchCentreData' in json_data:
                json_data = json_data['matchCentreData']
            if json_data is None or 'events' not in json_data:
                print(f"   ❌ No events in data")
                failed += 1
                processed_match_ids.add(match_id)
                continue
        except Exception as e:
            print(f"   ❌ Failed to parse JSON: {e}")
            failed += 1
            processed_match_ids.add(match_id)
            continue
        
        home_info = json_data.get('home', {})
        away_info = json_data.get('away', {})
        
        if not home_info or not away_info:
            print(f"   ❌ Missing team info")
            failed += 1
            processed_match_ids.add(match_id)
            continue
        
        # Process for each time period
        match_success = True
        for time_cutoff in time_periods:
            output_dir = base_data_dir / f"minute{time_cutoff}" / league_dir_name
            try:
                success = process_match_for_time(match_id, json_data, time_cutoff, output_dir, home_info, away_info)
                if success:
                    print(f"   ✅ minute{time_cutoff}")
                else:
                    print(f"   ⚠️ minute{time_cutoff} failed")
                    match_success = False
            except Exception as e:
                print(f"   ❌ minute{time_cutoff} error: {str(e)[:50]}")
                match_success = False
        
        if match_success:
            successful += 1
        else:
            failed += 1
        
        processed_match_ids.add(match_id)
    
    print(f"\n" + "=" * 60)
    print("Processing Complete!")
    print(f"=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Total processed: {len(processed_match_ids)}")


if __name__ == "__main__":
    main()
