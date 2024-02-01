from .train_episodes import simulate_episodes


def main():
    q_table = simulate_episodes(15000)
    print(q_table)


if __name__ == "__main__":
    main()
