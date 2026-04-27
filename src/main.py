# hugging face dataset for easier python integration than stanford palin text
from datasets import load_dataset


def main():
    ds = load_dataset("stanfordnlp/imdb")

    print(ds)
    print(ds["train"][0])
    print(ds["train"][-1])
    print(ds["test"][0])


if __name__ == "__main__":
    main()
