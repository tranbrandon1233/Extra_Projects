import random

class Property:
    def __init__(self, name, price=0, rent=0, buyable=False):
        self.name = name
        self.price = price
        self.owner = None
        self.rent = rent
        self.buyable = buyable

    def buy(self, player):
        if not self.buyable:
            return
        if player.money >= self.price:
            player.money -= self.price
            self.owner = player
            player.properties.append(self)
            print(f"\n{player.name} bought {self.name} for ${self.price}")
            print(f"{player.name} has ${player.money} remaining.")
            print(f"{player.name} now owns {', '.join([p.name for p in player.properties])}\n")
        else:
            print(f"{player.name} doesn't have enough money to buy {self.name}")

class Player:
    def __init__(self, name, money=1500):
        self.name = name
        self.money = money
        self.properties = []
        self.position = 0
        self.in_jail = False

    def move(self, dice_roll):
        self.position = (self.position + dice_roll) % len(board)
        print(f"{self.name} moved to space {self.position}")
        property = board[self.position]
        if property.owner and property.owner != self:
            self.pay_rent(property)
        elif not property.owner and self.money >= property.price:
            property.buy(self)

    def pay_rent(self, property):
        if property.name == "Community Chest":
            money_earned = random.randint(1, 100)
            print(f"\n{self.name} landed on Community Chest and got ${money_earned}!\n")
            self.money += money_earned
            print(f"{self.name} has ${self.money} remaining.\n")
            return
        if self.money >= property.rent:
            self.money -= property.rent
            property.owner.money += property.rent
            print(f"\n{self.name} lands on {property.name} and pays ${property.rent} to {property.owner.name}")
            print(f"{self.name} has ${self.money} remaining.\n")
        else:
            print(f"{self.name} is bankrupt! {property.owner.name} wins!")
            exit(1)
board = [
    Property("Go"),
    Property("Mediterranean Avenue", 60, 50, True),
    Property("Community Chest"),
    Property("Community Chest"),
    Property("Go To Jail"),
    Property("Community Chest"),
    Property("Community Chest"),
    Property("Go To Jail"),
    Property("Boardwalk", 200, 50, True),
    Property("Sidewalk Avenue", 60, 30, True),
    Property("Atlantic Street", 200, 50, True),
    Property("New Avenue", 60, 45, True),
    Property("Baltic Avenue", 200, 50, True),
    Property("Ramdom Avenue", 60, 35, True)

]

num_players = 2
players = [Player("Player " + str(i+1)) for i in range(num_players)]

current_player_index = 0

while len([p for p in players if p.money > 0]) > 1:
    current_player = players[current_player_index]

    dice_roll_1 = random.randint(1, 6)
    dice_roll_2 = random.randint(1, 6)
    dice_roll_total = dice_roll_1 + dice_roll_2

    current_player.move(dice_roll_total)
    
    if current_player.money >= 50 and current_player.in_jail:
        current_player.money -= 50
        current_player.in_jail = False
        print(f"{current_player.name} paid $50 to get out of jail.")

    if board[current_player.position].name == "Go" and not current_player.in_jail:
        current_player.money += 200 
        print(f"{current_player.name} collected £200 from passing Go.")
    elif board[current_player.position].name == "Go To Jail":
        if current_player.money >= 50:
            current_player.money -= 50
            print(f"\n{current_player.name} went to jail and paid $50 to get out.\n")
        else:
            current_player.in_jail = True  
            print(f"\n{current_player.name} went to jail and needs to pay $50 to get out.\n")


    current_player_index = (current_player_index + 1) % num_players

    input(f"Press Enter to continue for {current_player.name}'s turn...")

print(f"\nGame Over! The winner is {players[current_player_index].name} with ${players[current_player_index].money}!") 