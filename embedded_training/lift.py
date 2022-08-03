#!/usr/bin/env python3
from enum import Enum

class Lift_state(Enum):
    Init = 1
    Close_stopped = 2
    Close_moving = 3
    Open_stopped = 4

def close_stop_entry(target_level):
    print(f'lift is stopped at level {target_level}, door closed')

def close_stop_exit():
    return

def close_moving_entry(target_level):
    print(f'lift starts moving to level {target_level}')

def close_moving_loop(curr_level, target_level):
    return curr_level == target_level

def close_moving_exit(target_level):
    print(f'lift arrived level {target_level}')

def open_stopped_entry():
    print('door opened')

def open_stopped_exit():
    print('door closed')

def main():
    lift = Lift_state.Init
    curr = level = 1

    while True:
        if lift == Lift_state.Init:
            close_stop_entry(level)
            lift = Lift_state.Close_stopped
        elif lift == Lift_state.Close_stopped:
            inp = input("Please input number 1/2/3/4 for target level,'C' for closing door or 'O' for opening door: 1/2/3/4/C/O ").upper()
            if inp in [str(i + 1) for i in range(4)]:
                if curr == int(inp):
                    print(f'lift is already stopped at level {curr}')
                else:
                    level = int(inp)
                    close_stop_exit()
                    close_moving_entry(level)
                    lift = Lift_state.Close_moving
            elif inp in 'OC':
                if inp == 'O':
                    close_stop_exit()
                    open_stopped_entry()
                    lift = Lift_state.Open_stopped
                elif inp == 'C':
                    print('door remains closed')
        elif lift == Lift_state.Close_moving:
            if close_moving_loop(curr, level):
                close_moving_exit(level)
                close_stop_entry(level)
                lift = Lift_state.Close_stopped
            else:
                # simulate lift moving phase
                curr += 1 if level > curr else -1
                print(f'lift moved to level {curr}')
        elif lift == Lift_state.Open_stopped:
            inp = input("Please press 'C' to close door, no other action is allowed: ").upper()
            if inp == 'C':
                open_stopped_exit()
                close_stop_entry(level)
                lift = Lift_state.Close_stopped
            else:
                print("please press 'C' to close door first")
        else:
            print(f'unknown lift state: {lift}')

if __name__ == '__main__':
    main()