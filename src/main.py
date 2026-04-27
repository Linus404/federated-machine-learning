# hugging face dataset for easier python integration than stanford palin text
from datasets import load_dataset, DatasetDict


def main():
    ds = load_dataset("stanfordnlp/imdb")

    print(ds)
    print(ds["train"][0])
    print(ds["train"][-1])
    print(ds["test"][0])


    ShuffeledDataset=ds.shuffle(seed=67)

    Client1Datensatz:DatasetDict=ShuffeledDataset["train"].select(range(0,5000))

    Client3Datensatz: DatasetDict = ShuffeledDataset["train"].select(range(5000, 10000))

    Client4Datensatz: DatasetDict = ShuffeledDataset["train"].select(range(10000, 15000))

    Client5Datensatz: DatasetDict = ShuffeledDataset["train"].select(range(15000, 20000))

    for i in Client1Datensatz:
        print(i)



if __name__ == "__main__":
    main()
