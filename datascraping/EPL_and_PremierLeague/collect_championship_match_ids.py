import soccerdata as sd
import json
import pandas as pd
from pathlib import Path
import sys

def main():
    print("============================================================")
    print("Collecting Match IDs from England Championship (Native Mode)")
    print("Target: Seasons 20/21, 21/22, 22/23")
    print("============================================================")

    # 1. Initialize Soccerdata with our custom league
    # Note: We must have 'ENG-Championship' in league_dict.json for this to work
    try:
        # Seasons format in soccerdata: '2021' = 2020-2021. 
        # We need 20/21, 21/22, 22/23 -> '2021', '2122', '2223'
        seasons = ['2021', '2122', '2223']
        
        print(f"🚀 Initializing soccerdata for {seasons}...")
        ws = sd.WhoScored(leagues="ENG-Championship", seasons=seasons, headless=False)
        
        print("📖 Reading schedule (this involves scraping, please wait)...")
        schedule = ws.read_schedule()
        
        print(f"✅ Scrape complete. shape: {schedule.shape}")
        
    except Exception as e:
        print(f"❌ Failed to use soccerdata: {e}")
        print("   Make sure /Users/merthandurdag/soccerdata/config/league_dict.json exists and is correct.")
        sys.exit(1)

    # 2. Process Data
    matches = []
    if not schedule.empty:
        # Reset index to get league/season/game_id as columns if they are in index
        schedule = schedule.reset_index()
        
        for _, row in schedule.iterrows():
            # Extract fields
            try:
                match_id = int(row['game_id'])
                season_str = str(row['season']) # e.g. "20-21" or "2020-2021"
                
                # Format scores
                h_score = 0
                a_score = 0
                try:
                    h_score = int(row['home_score']) if pd.notna(row['home_score']) else 0
                    a_score = int(row['away_score']) if pd.notna(row['away_score']) else 0
                except: pass
                
                # Date
                date_str = str(row['date']) if 'date' in row else ""

                matches.append({
                    'match_id': match_id,
                    'league': 'ENG-Championship',
                    'season': season_str,
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_score': h_score,
                    'away_score': a_score,
                    'date': date_str
                })
            except Exception as e:
                print(f"   ⚠️ Skipping row: {e}")
                
    # 3. Deduplicate (just in case)
    seen = set()
    unique_matches = []
    for m in matches:
        if m['match_id'] not in seen:
            seen.add(m['match_id'])
            unique_matches.append(m)
            
    print(f"📊 Collected {len(unique_matches)} unique matches.")
    
    # 4. Save
    script_dir = Path(__file__).parent
    output_file = script_dir / f"match_ids_championship_{len(unique_matches)}.json"
    
    with open(output_file, 'w') as f:
        json.dump(unique_matches, f, indent=2)
            
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    main()
