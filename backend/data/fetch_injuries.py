"""
Fetch NFL injury data from multiple sources:
1. ESPN web scraping
2. Sleeper API (free, no auth)

Run weekly to get current injury reports.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def fetch_sleeper_injuries():
    """
    Fetch player injury data from Sleeper API.
    Free, no auth required.
    Docs: https://docs.sleeper.com/
    """
    print("Fetching from Sleeper API...")

    # Get all NFL players (includes injury status)
    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()

    players = response.json()
    print(f"Retrieved {len(players)} players from Sleeper")

    # Extract injury info
    injuries = []
    for player_id, player in players.items():
        if player.get('injury_status') or player.get('injury_body_part'):
            injuries.append({
                'player_id': player_id,
                'full_name': player.get('full_name'),
                'first_name': player.get('first_name'),
                'last_name': player.get('last_name'),
                'team': player.get('team'),
                'position': player.get('position'),
                'injury_status': player.get('injury_status'),  # Out, Doubtful, Questionable, IR, etc.
                'injury_body_part': player.get('injury_body_part'),
                'injury_start_date': player.get('injury_start_date'),
                'injury_notes': player.get('injury_notes'),
                'status': player.get('status'),  # Active, Inactive, IR
                'espn_id': player.get('espn_id'),
                'yahoo_id': player.get('yahoo_id'),
                'gsis_id': player.get('gsis_id'),
            })

    df = pd.DataFrame(injuries)
    print(f"Found {len(df)} players with injury info")

    return df


def scrape_espn_injuries():
    """
    Scrape injury data from ESPN NFL injuries page.
    """
    print("Scraping ESPN injuries page...")

    url = "https://www.espn.com/nfl/injuries"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    injuries = []

    # Find all team sections
    team_sections = soup.find_all('div', class_='ResponsiveTable')

    for section in team_sections:
        # Get team name from header
        team_header = section.find_previous('div', class_='Table__Title')
        team_name = team_header.get_text(strip=True) if team_header else 'Unknown'

        # Find all rows in the table
        rows = section.find_all('tr', class_='Table__TR')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4:
                # Extract player info
                name_cell = cells[0]
                player_link = name_cell.find('a')
                player_name = player_link.get_text(strip=True) if player_link else name_cell.get_text(strip=True)

                injuries.append({
                    'team': team_name,
                    'player_name': player_name,
                    'position': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                    'injury_date': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                    'injury_status': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                    'injury_type': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                })

    df = pd.DataFrame(injuries)
    print(f"Scraped {len(df)} injury records from ESPN")

    return df


def fetch_espn_api_injuries():
    """
    Fetch injuries from ESPN's API endpoints.
    """
    print("Fetching from ESPN API...")

    # NFL team IDs
    teams = {
        'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
        'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
        'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
        'LAR': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
        'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
        'TEN': 10, 'WAS': 28
    }

    all_injuries = []

    for team_abbr, team_id in teams.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/injuries"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()

                # Parse injuries
                for item in data.get('items', []):
                    athlete = item.get('athlete', {})
                    injuries_list = item.get('injuries', [])

                    for injury in injuries_list:
                        all_injuries.append({
                            'team': team_abbr,
                            'player_id': athlete.get('id'),
                            'full_name': athlete.get('displayName'),
                            'position': athlete.get('position', {}).get('abbreviation'),
                            'jersey': athlete.get('jersey'),
                            'injury_type': injury.get('type', {}).get('description'),
                            'injury_location': injury.get('location', {}).get('description'),
                            'injury_detail': injury.get('detail', {}).get('description'),
                            'injury_status': injury.get('status'),
                            'injury_date': injury.get('date'),
                        })
        except Exception as e:
            print(f"Error fetching {team_abbr}: {e}")

    df = pd.DataFrame(all_injuries)
    print(f"Retrieved {len(df)} injuries from ESPN API")

    return df


def main():
    """Fetch injuries from all sources and save."""

    print("="*80)
    print("FETCHING NFL INJURY DATA")
    print("="*80 + "\n")

    # Use relative path from script location
    outputs_dir = Path(__file__).parent.parent.parent / 'inputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sleeper API (best source - free, comprehensive)
    sleeper_df = fetch_sleeper_injuries()
    if len(sleeper_df) > 0:
        sleeper_df.to_csv(outputs_dir / 'injuries_sleeper_current.csv', index=False)
        print(f"Saved: injuries_sleeper_current.csv ({len(sleeper_df)} records)\n")

    # 2. ESPN API
    espn_api_df = fetch_espn_api_injuries()
    if len(espn_api_df) > 0:
        espn_api_df.to_csv(outputs_dir / 'injuries_espn_api_current.csv', index=False)
        print(f"Saved: injuries_espn_api_current.csv ({len(espn_api_df)} records)\n")

    # 3. ESPN Web Scrape (backup)
    # espn_web_df = scrape_espn_injuries()
    # if len(espn_web_df) > 0:
    #     espn_web_df.to_csv(outputs_dir / 'injuries_espn_web_current.csv', index=False)

    # Summary
    print("\n" + "="*80)
    print("INJURY DATA FETCH COMPLETE")
    print("="*80)

    print("\nFiles created:")
    print(f"  - injuries_sleeper_current.csv: {len(sleeper_df)} players")
    print(f"  - injuries_espn_api_current.csv: {len(espn_api_df)} injuries")

    print("\nKey injury statuses from Sleeper:")
    if len(sleeper_df) > 0:
        print(sleeper_df['injury_status'].value_counts().to_string())

    print("\n" + "="*80)

    return sleeper_df, espn_api_df


if __name__ == "__main__":
    sleeper_df, espn_df = main()
