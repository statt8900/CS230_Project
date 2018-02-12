#External modules
import os



def test_get_data(limit = 20):
    from CS230_Project import CNN_input
    return CNN_input.CNNInputDataset(limit = limit)


def main():
    return test_get_data()


if __name__ == '__main__':
    CNN_Input = main()
