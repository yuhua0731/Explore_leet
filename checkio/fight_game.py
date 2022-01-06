#!/usr/bin/env python3
import math


class Warrior:
    """
    base Warrior class
    - only has attack and health as his initial data
    - allows user to check if he is alive or not
    - two functions: hit & loss
    """

    def __init__(self, health=50, attack=5):
        self.health = health
        self.attack = attack

    @property
    def is_alive(self):
        return self.health > 0

    def hit(self, other):
        other.loss(self.attack)

    def loss(self, attack):
        self.health -= attack
        return attack


class Knight(Warrior):
    """
    Knight: stronger than basic warrior
    """

    def __init__(self):
        super().__init__(attack=7)


class Defender(Warrior):
    """
    Defender:
    - additional initial property: defense
    - overwrite loss function: consider his defense when got hit
    """

    def __init__(self):
        super().__init__(health=60, attack=3)
        self.defense = 2

    def loss(self, attack):
        self.health -= max(0, attack - self.defense)
        return max(0, attack - self.defense)


class Vampire(Warrior):
    """
    Vampire:
    When the Vampire hits the other unit, he restores his health by +50% of the dealt damage (enemy defense makes the dealt damage value lower).
    The basic parameters of the Vampire:
    health = 40
    attack = 4
    vampirism = 50%
    """

    def __init__(self):
        super().__init__(health=40, attack=4)
        self.vampirism = 0.5

    def hit(self, other):
        self.health += math.floor(other.loss(self.attack) * self.vampirism)


class Lancer(Warrior):
    """
    Lancer:
    when he hits the other unit, he also deals a 50% of the deal damage to the enemy unit, standing behind the firstly assaulted one (enemy defense makes the deal damage value lower - consider this).
    The basic parameters of the Lancer:
    health = 50
    attack = 6
    """

    def __init__(self):
        super().__init__(attack=6)


def fight(unit_1, unit_2):
    while True:
        unit_1.hit(unit_2)
        if not unit_2.is_alive:
            return True
        unit_2.hit(unit_1)
        if not unit_1.is_alive:
            return False


class Army:
    def __init__(self):
        self.army = list()
        self.first_fighter_idx = 0

    def add_units(self, unit_class, amount):
        self.army += [unit_class() for _ in range(amount)]

    def lose_fighter(self):
        self.first_fighter_idx += 1

    def first_fighter(self):
        return self.army[self.first_fighter_idx] if self.first_fighter_idx < len(self.army) else None

    @property
    def is_alive(self):
        return self.first_fighter_idx < len(self.army)


class Battle:
    def fight(self, army_1: Army, army_2: Army):
        while army_1.is_alive and army_2.is_alive:
            duel = fight(army_1.get_first(), army_2.get_first())
            if duel:
                army_2.lose_fighter()
            else:
                army_1.lose_fighter()
        return army_1.is_alive


if __name__ == '__main__':
    # These "asserts" using only for self-checking and not necessary for auto-testing

    # fight tests
    chuck = Warrior()
    bruce = Warrior()
    carl = Knight()
    dave = Warrior()
    mark = Warrior()
    bob = Defender()
    mike = Knight()
    rog = Warrior()
    lancelot = Defender()
    eric = Vampire()
    adam = Vampire()
    richard = Defender()
    ogre = Warrior()
    freelancer = Lancer()
    vampire = Vampire()

    assert fight(chuck, bruce) == True
    assert fight(dave, carl) == False
    assert chuck.is_alive == True
    assert bruce.is_alive == False
    assert carl.is_alive == True
    assert dave.is_alive == False
    assert fight(carl, mark) == False
    assert carl.is_alive == False
    assert fight(bob, mike) == False
    assert fight(lancelot, rog) == True
    assert fight(eric, richard) == False
    assert fight(ogre, adam) == True
    assert fight(freelancer, vampire) == True
    assert freelancer.is_alive == True

    # battle tests
    my_army = Army()
    my_army.add_units(Defender, 2)
    my_army.add_units(Vampire, 2)
    my_army.add_units(Lancer, 4)
    my_army.add_units(Warrior, 1)

    enemy_army = Army()
    enemy_army.add_units(Warrior, 2)
    enemy_army.add_units(Lancer, 2)
    enemy_army.add_units(Defender, 2)
    enemy_army.add_units(Vampire, 3)

    army_3 = Army()
    army_3.add_units(Warrior, 1)
    army_3.add_units(Lancer, 1)
    army_3.add_units(Defender, 2)

    army_4 = Army()
    army_4.add_units(Vampire, 3)
    army_4.add_units(Warrior, 1)
    army_4.add_units(Lancer, 2)

    battle = Battle()

    assert battle.fight(my_army, enemy_army) == True
    assert battle.fight(army_3, army_4) == False
    print("Coding complete? Let's try tests!")
