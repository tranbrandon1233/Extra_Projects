import itertools

# Define player data structure
class Player:
    def __init__(self, name, positions):
        self.name = name
        self.positions = positions

# Define team structure
team_structure = {
    "HOK": 1,
    "MID": 3,
    "EDG": 2,
    "HLF": 2,
    "CTR": 2,
    "WFB": 3,
    "INT": 4
}

# Example squad of 21 players with their positions
squad = [
    Player("Player1", ["HOK"]),
    Player("Player2", ["HOK", "MID"]),
    Player("Player3", ["MID"]),
    Player("Player4", ["MID"]),
    Player("Player5", ["MID"]),
    Player("Player6", ["MID", "EDG"]),
    Player("Player7", ["EDG"]),
    Player("Player8", ["EDG"]),
    Player("Player9", ["HLF"]),
    Player("Player10", ["HLF"]),
    Player("Player11", ["HLF", "CTR"]),
    Player("Player12", ["CTR"]),
    Player("Player13", ["CTR"]),
    Player("Player14", ["WFB"]),
    Player("Player15", ["WFB"]),
    Player("Player16", ["WFB"]),
    Player("Player17", ["HOK", "INT"]),
    Player("Player18", ["MID", "INT"]),
    Player("Player19", ["EDG", "INT"]),
    Player("Player20", ["HLF", "INT"]),
    Player("Player21", ["CTR", "INT"]),
]

def create_team(squad, team_structure):
    """
    Creates a valid NRL fantasy team based on the given squad and team structure.

    Args:
        squad: A list of Player objects representing the available players.
        team_structure: A dictionary defining the required number of players for each position.

    Returns:
        A dictionary representing the team with positions as keys and lists of player names as values,
        or None if no valid team can be formed.
    """

    # Create a dictionary to store players by position
    players_by_position = {}
    for position in team_structure:
        players_by_position[position] = [player for player in squad if position in player.positions]

    # Iterate through all possible combinations of players for each position
    for team_combination in itertools.product(*players_by_position.values()):
        # Check if the combination satisfies the team structure
        team = {}
        for position, num_players in team_structure.items():
            team[position] = []
            for player in team_combination:
                if position in player.positions and len(team[position]) < num_players:
                    team[position].append(player.name)
                    break
        if all(len(players) == num_players for position, players in team.items()):
            return team

    # No valid team found
    return None

# Create and print the team
team = create_team(squad, team_structure)
if team:
    print("Team:")
    for position, players in team.items():
        print(f"{position}: {players}")
else:
    print("No valid team can be formed with the given squad.")