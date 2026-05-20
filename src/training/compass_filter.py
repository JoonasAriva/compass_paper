import pandas as pd


class CompassFilter:
    def __init__(self, df_train_path: str, df_test_path: str):
        super().__init__()

        self.df = pd.concat([pd.read_csv(df_train_path), pd.read_csv(df_test_path)])

        self.filter_min = 5.5
        self.filter_max = 25

        filtered = self.df[
            self.df["compass_scores"].between(self.filter_min, self.filter_max)
        ]
        # Take first and last passing slice index per scan
        self.slice_lookup = (
            filtered.groupby("scan_id")["slice_nr"]
            .agg(start_idx="min", end_idx="max")
        )
        # Result is indexed by scan_path for O(1) lookup

        print("Compass cutoffs: ", self.filter_min, self.filter_max)

    def get_indexes(self, case_id: str):

        if case_id not in self.slice_lookup.index:
            return None, None

        row = self.slice_lookup.loc[case_id]
        start, end = int(row["start_idx"]), int(row["end_idx"]) + 1  # +1 for inclusive end

        return start, end
