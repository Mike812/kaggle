import argparse
import os

import pandas as pd

data_path = "../data/"

for dir_name, _, file_names in os.walk(data_path):
    for filename in file_names:
        print(os.path.join(dir_name, filename))

players_and_years = pd.read_csv(data_path + "players(man).csv")
players_tournament_results = pd.read_csv(data_path + "players_tournament(man).csv")
players_tournament_detailed = pd.read_csv(data_path + "raw_kaggle.csv")
players_tournament_serve_data = pd.read_csv(data_path + "serve_kaggle.csv")
players_tournament_return_data = pd.read_csv(data_path + "return_kaggle.csv")


def query_players_tournament_detailed(tennis_player, tournament, year):
    df_result = players_tournament_detailed[players_tournament_detailed.Name.str.contains(tennis_player)
                                         & players_tournament_detailed.Tournament.str.contains(tournament)
                                         & players_tournament_detailed.Date.str.contains(year)]
    return df_result


def main():
    parser = argparse.ArgumentParser(description="This is a command line interface for tennis stats")

    parser.add_argument("--tennis_player", help="Name of the tennis player")

    parser.add_argument("--tournament", help="Name of the tournament")

    parser.add_argument("--year", help="Year the tournament took place")

    args = parser.parse_args()
    tennis_player = args.tennis_player
    tournament = args.tournament
    year = args.year

    print("These are the tennis stats of " + args.tennis_player + " at the "
          + args.tournament + " in " + args.year + ": ")

    df_result = query_players_tournament_detailed(tennis_player=tennis_player, tournament=tournament, year=year)

    print(df_result.head())


if __name__ == "__main__":
    main()
