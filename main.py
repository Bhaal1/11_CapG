import pandas as pd


def load_data():
    df = pd.read_csv('input/Recruiting_Task_InputData.csv')
    print(df.head())


if __name__ == '__main__':
    load_data()


