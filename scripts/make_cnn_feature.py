from glob import glob
import pandas as pd


def main():

    paths = glob("./output/simple_3dcnn/fold*/result/valid_result_all.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=0)
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/3dcnn_feature.csv", index=False
    )

    paths = glob("./output/simple_3dcnn/fold*/result/test_result.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/3dcnn_feature_test.csv", index=False
    )

    paths = glob("./output/2plus1dcnn_resnet10/fold*/result/valid_result_all.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=0)
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/2plus1dcnn_feature.csv", index=False
    )

    paths = glob("./output/2plus1dcnn_resnet10/fold*/result/test_result.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=1)
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/2plus1dcnn_feature_test.csv", index=False
    )


    paths = glob("./output/gin/fold*/result/valid_result_all.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=0)
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/gnn_feature.csv", index=False
    )

    paths = glob("./output/gin/fold*/result/test_result.csv")
    df = pd.concat([pd.read_csv(path) for path in paths], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df[[col for col in df.columns if col.startswith("feature")] + ["Id"]].to_csv(
        "./input/gnn_feature_test.csv", index=False
    )


if __name__ == "__main__":
    main()
