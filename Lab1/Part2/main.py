import one
import three
import two

if __name__ == "__main__":
    while True:
        try:
            option = int(
                input("""Please choose 1,2, or 3\n
                    1. Run part 1.\n
                    2. Run part 2.\n
                    3. Run part 3.\n
                    Any other key: Exit application.\n""")
            )
            match option:
                case 1:
                    one.main()
                case 2:
                    two.main()
                case 3:
                    three.main()
                case _:
                    print("Exited")
                    break
        except ValueError:
            print("Exited")
            break
