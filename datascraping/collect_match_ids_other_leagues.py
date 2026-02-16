"""
Collect Match IDs for Other Leagues - 22/23 Season Only

Collects match IDs ONLY from the 2022-23 season for:
- Serie A (Italy)
- La Liga (Spain)  
- Bundesliga (Germany)
- Ligue 1 (France)

This matches the paper's statement: "We collected additional match data from 
Serie A (Italy), La Liga (Spain), Bundesliga (Germany), and Ligue 1 (France) 
for the 22/23 season."

Usage:
    python collect_match_ids_other_leagues.py
"""

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
print("Collecting Match IDs for Other Leagues - 22/23 Season ONLY")
print("=" * 60)

# Other leagues (NOT Premier League) - 22/23 season only
# Soccerdata: 2022 = 2022-23 season (August 2022 - May 2023)
leagues = {
    "ESP-La Liga": [2022],      # La Liga 22/23
    "ITA-Serie A": [2022],      # Serie A 22/23
    "GER-Bundesliga": [2022],   # Bundesliga 22/23
    "FRA-Ligue 1": [2022],      # Ligue 1 22/23
}

all_matches = []

for league_name, seasons in leagues.items():
    league_matches = []
    
    for season in seasons:
        season_label = f"{season}-{str(season+1)[-2:]}"  # 2022 -> "2022-23"
        print(f"\n📊 Processing {league_name} {season_label} season...")
        
        try:
            ws = sd.WhoScored(leagues=league_name, seasons=season, headless=True)
            
            # Load schedule
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
                # DataFrame check
                if not isinstance(schedule, pd.DataFrame):
                    print(f"   ⚠️  Schedule is not a DataFrame, converting...")
                    try:
                        schedule = pd.DataFrame(schedule)
                    except:
                        schedule = None
                        print(f"   ❌ Could not convert to DataFrame")
                
                if schedule is not None and len(schedule) > 0:
                    # status == 6 = played match
                    if 'status' in schedule.columns:
                        played = schedule[schedule['status'] == 6].copy()
                    else:
                        played = schedule.copy()
                        print(f"   ⚠️  No 'status' column found, using all matches")
                    
                    if len(played) > 0:
                        if 'game_id' in played.columns:
                            # Sort by date
                            if 'date' in played.columns:
                                played = played.sort_values('date', ascending=True)
                            
                            # Collect ALL played matches from this season
                            for idx, (_, match_info) in enumerate(played.iterrows()):
                                match_id = match_info['game_id']
                                
                                match_data = {
                                    'match_id': int(match_id) if pd.notna(match_id) else None,
                                    'league': league_name,
                                    'season': season,
                                    'season_label': season_label,
                                    'home_team': match_info.get('home_team', ''),
                                    'away_team': match_info.get('away_team', ''),
                                    'home_score': match_info.get('home_score', 0),
                                    'away_score': match_info.get('away_score', 0),
                                    'date': str(match_info.get('date', '')) if 'date' in match_info.index else '',
                                }
                                
                                if match_data['match_id'] is not None:
                                    league_matches.append(match_data)
                            
                            print(f"   ✅ Collected {len(league_matches)} matches from {season_label}")
                        else:
                            print(f"   ⚠️  Could not find 'game_id' column in schedule")
                            print(f"   Available columns: {list(schedule.columns)}")
                    else:
                        print(f"   ⚠️  No played matches found in {season_label} schedule")
                else:
                    print(f"   ⚠️  Schedule is empty for {season_label}")
            else:
                print(f"   ⚠️  Could not load schedule for {season_label}, skipping")
            
            # Close browser
            try:
                if hasattr(ws, '_driver') and ws._driver:
                    ws._driver.quit()
            except:
                pass
            
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error processing {league_name} {season_label}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:200]}")
            continue
    
    all_matches.extend(league_matches)

print(f"\n" + "=" * 60)
print(f"Total matches collected (before de-dup): {len(all_matches)}")

# Duplicate check
seen_ids = set()
unique_matches = []
for match in all_matches:
    match_id = match['match_id']
    if match_id not in seen_ids:
        seen_ids.add(match_id)
        unique_matches.append(match)
    else:
        print(f"   ⚠️  Duplicate removed: Match {match_id}")

all_matches = unique_matches
print(f"After removing duplicates: {len(all_matches)} unique matches")

script_dir = Path(__file__).parent
output_file = script_dir / f"match_ids_other_leagues_2223.json"
with open(output_file, 'w') as f:
    json.dump(all_matches, f, indent=2)

print(f"\n✅ Match IDs saved to: {output_file}")
print(f"   Total matches: {len(all_matches)}")

# Summary by league
league_counts = {}
for match in all_matches:
    league = match['league']
    league_counts[league] = league_counts.get(league, 0) + 1

print("\n📊 Summary by league (22/23 season only):")
for league, count in sorted(league_counts.items()):
    print(f"   {league}: {count} matches")

print(f"\n🎯 Expected: ~380 matches per league (full season)")
print(f"   Actual total: {len(all_matches)} matches")
