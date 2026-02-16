import sys
import json
from pathlib import Path
import time
import pandas as pd

try:
    import soccerdata as sd
except ImportError:
    print("soccerdata not installed")
    sys.exit(1)

print("=" * 60)
print("Collecting Match IDs from Premier League")
print("Target: ALL played matches per league (20/21, 21/22, 22/23)")
print("=" * 60)

# EPL seasons (20/21, 21/22, 22/23) — explicit season codes to avoid ambiguity
leagues = {
    "ENG-Premier League": ["2022-2023", "2021-2022", "2020-2021"],
}

all_matches = []

for league_name, seasons in leagues.items():
    league_matches = []
    # Önce belirtilen tüm sezonlardan maçları topla
    for season in seasons:
        # Sezon etiketi: string kod ise doğrudan kullan
        season_label = season if isinstance(season, str) else f"{season}-{str(season+1)[-2:]}"  # 2024 -> "2024-25"
        print(f"\n📊 Processing {league_name} {season_label} season ({season})...")
        try:
            ws = sd.WhoScored(leagues=league_name, seasons=season, headless=False)
            
            # GitHub koduna göre read_schedule() kullan
            schedule = None
            try:
                schedule = ws.read_schedule()
                if schedule is not None and len(schedule) > 0:
                    print(f"   ✅ Schedule loaded: {len(schedule)} matches found")
                else:
                    print(f"   ⚠️  Schedule is empty")
            except Exception as e1:
                print(f"   ⚠️  read_schedule() failed: {str(e1)[:100]}")
                schedule = None
            
            if schedule is not None and len(schedule) > 0:
                # DataFrame kontrolü
                if not isinstance(schedule, pd.DataFrame):
                    print(f"   ⚠️  Schedule is not a DataFrame, converting...")
                    try:
                        schedule = pd.DataFrame(schedule)
                    except:
                        schedule = None
                        print(f"   ❌ Could not convert to DataFrame")
                
                if schedule is not None and len(schedule) > 0:
                    # GitHub koduna göre: status == 6 = oynanmış maç
                    # Sadece oynanmış maçları al
                    if 'status' in schedule.columns:
                        played = schedule[schedule['status'] == 6].copy()
                    else:
                        # Status kolonu yoksa, tüm maçları al (zaten filtrelenmiş olabilir)
                        played = schedule.copy()
                        print(f"   ⚠️  No 'status' column found, using all matches")
                    
                    if len(played) > 0:
                        # GitHub koduna göre: game_id kolonu var
                        if 'game_id' in played.columns:
                            # Tarihe göre sırala (en yeni önce) - date kolonu var
                            if 'date' in played.columns:
                                played = played.sort_values('date', ascending=False)
                            
                            # Bu sezondan tüm oynanmış maçları ekle
                            for idx, (_, match_info) in enumerate(played.iterrows()):
                                match_id = match_info['game_id']
                                
                                match_data = {
                                    'match_id': int(match_id) if pd.notna(match_id) else None,
                                    'league': league_name,
                                    'season': season,
                                    'home_team': match_info.get('home_team', ''),
                                    'away_team': match_info.get('away_team', ''),
                                    'home_score': match_info.get('home_score', 0),
                                    'away_score': match_info.get('away_score', 0),
                                    'date': str(match_info.get('date', '')) if 'date' in match_info.index else '',
                                }
                                
                                if match_data['match_id'] is not None:
                                    league_matches.append(match_data)
                            
                            print(f"   ✅ Collected {len([m for m in league_matches if m.get('season') == season])} matches from {season_label} (total: {len(league_matches)})")
                        else:
                            print(f"   ⚠️  Could not find 'game_id' column in schedule")
                            print(f"   Available columns: {list(schedule.columns)}")
                    else:
                        print(f"   ⚠️  No played matches found in {season_label} schedule")
                else:
                    print(f"   ⚠️  Schedule is empty for {season_label}")
            else:
                print(f"   ⚠️  Could not load schedule for {season_label}, skipping")
            
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error processing {league_name} {season_label}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:200]}")
            continue
    
    # Tüm sezonlardan maçları topladıktan sonra, lig içi sıralama (limit yok)
    if len(league_matches) > 0:
        # Tarihe göre sırala (en yeni önce)
        league_matches_sorted = sorted(
            league_matches,
            key=lambda x: x.get('date', ''),
            reverse=True
        )
        league_matches = league_matches_sorted
        print(f"   📦 Total collected from {league_name}: {len(league_matches)} matches (sorted by date, most recent first)")
    else:
        print(f"   ⚠️  No matches collected for {league_name}")
    
    all_matches.extend(league_matches)

print(f"\n" + "=" * 60)
print(f"Total matches collected (before de-dup): {len(all_matches)}")

# Duplicate kontrolü
seen_ids = set()
unique_matches = []
for match in all_matches:
    match_id = match['match_id']
    if match_id not in seen_ids:
        seen_ids.add(match_id)
        unique_matches.append(match)
    else:
        print(f"   ⚠️  Duplicate removed: Match {match_id} ({match.get('home_team')} vs {match.get('away_team')})")

all_matches = unique_matches
print(f"After removing duplicates: {len(all_matches)} unique matches")

script_dir = Path(__file__).parent
output_file = script_dir / f"match_ids_{len(all_matches)}.json"
with open(output_file, 'w') as f:
    json.dump(all_matches, f, indent=2)

print(f"✅ Match IDs saved to: {output_file}")
print(f"   Total matches: {len(all_matches)}")

# Lig bazında özet
league_counts = {}
for match in all_matches:
    league = match['league']
    league_counts[league] = league_counts.get(league, 0) + 1

print("\n📊 Summary by league:")
for league, count in sorted(league_counts.items()):
    print(f"   {league}: {count} matches")
    
print(f"\n🎯 Target: ALL played matches from 20/21, 21/22, 22/23")
print(f"   Actual total: {len(all_matches)} matches")
