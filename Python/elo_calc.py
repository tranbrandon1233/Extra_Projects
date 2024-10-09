import pandas as pd

def calculate_elo_ratings(df, initial_elo=1500, k_factor=32):
    """
    Calculates stable Elo ratings for players in a dataframe of games.

    Args:
        df (pd.DataFrame): Dataframe with columns 'player1', 'player2', and 'winner'.
                            Each row represents a game, and 'winner' indicates the winner 
                            (either 'player1' or 'player2').
        initial_elo (int, optional): Starting Elo rating for all players. Defaults to 1500.
        k_factor (int, optional): K-factor determines the impact of a single game. 
                                   Higher values mean larger rating changes. Defaults to 32.

    Returns:
        dict: A dictionary where keys are player names and values are their calculated Elo ratings.
    """

    # Create a dictionary to store Elo ratings
    elo_ratings = {}

    # Function to calculate the expected score for a player
    def expected_score(player_a_elo, player_b_elo):
        return 1 / (1 + 10 ** ((player_b_elo - player_a_elo) / 400))

    # Function to update Elo ratings after a game
    def update_elo(winner_elo, loser_elo):
        expected_win = expected_score(winner_elo, loser_elo)
        winner_elo += k_factor * (1 - expected_win)
        loser_elo -= k_factor * (1 - expected_win)
        return winner_elo, loser_elo

    # Initialize Elo ratings
    all_players = set(df['player1']).union(set(df['player2']))
    for player in all_players:
        elo_ratings[player] = initial_elo

    # Iterate through the games multiple times to achieve stability
    num_iterations = 10  # Adjust this value for desired stability
    for _ in range(num_iterations):
        df_shuffled = df.sample(frac=1).reset_index(drop=True)  # Shuffle the games
        for _, row in df_shuffled.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']

            if winner == 'player1':
                elo_ratings[player1], elo_ratings[player2] = update_elo(elo_ratings[player1], elo_ratings[player2])
            else:
                elo_ratings[player2], elo_ratings[player1] = update_elo(elo_ratings[player2], elo_ratings[player1])

    return elo_ratings

# Example usage:
data = {'player1': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
        'player2': ['Bob', 'Charlie', 'Alice', 'Charlie', 'Alice'],
        'winner': ['Alice', 'Bob', 'Alice', 'Alice', 'Bob']}
games_df = pd.DataFrame(data)

elo_ratings = calculate_elo_ratings(games_df)
print(elo_ratings) 