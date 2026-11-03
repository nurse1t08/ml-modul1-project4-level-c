import unittest
import numpy as np
import pandas as pd
from src.hotel_project import (MODEL_FEATURES, POST_BOOKING, TARGET, feature_frame,
                               split_data, make_pipeline, threshold_table,
                               choose_threshold, group_errors)
from generate_demo import generate

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate(n=360, seed=3)

    def test_demo_is_32_columns(self):
        self.assertEqual(self.df.shape[1], 32)

    def test_leakage_features_are_excluded(self):
        X = feature_frame(self.df)
        self.assertNotIn(TARGET, X.columns)
        self.assertFalse(set(POST_BOOKING) & set(X.columns))
        self.assertEqual(X.columns.tolist(), MODEL_FEATURES)

    def test_splits_disjoint(self):
        a, b, c = split_data(self.df)
        self.assertFalse(set(a.index) & set(b.index))
        self.assertFalse(set(a.index) & set(c.index))
        self.assertFalse(set(b.index) & set(c.index))
        self.assertEqual(len(a) + len(b) + len(c), len(self.df))

    def test_duplicate_features_never_cross_splits(self):
        duplicate = self.df.iloc[:25].copy()
        duplicate.index = range(10000, 10025)
        combined = pd.concat([self.df, duplicate])
        a, b, c = split_data(combined)
        hashes = [set(pd.util.hash_pandas_object(feature_frame(x), index=False))
                  for x in (a, b, c)]
        self.assertFalse(hashes[0] & hashes[1])
        self.assertFalse(hashes[0] & hashes[2])
        self.assertFalse(hashes[1] & hashes[2])

    def test_unknown_category_and_na(self):
        train, val, _ = split_data(self.df)
        X = feature_frame(train)
        Y = feature_frame(val)
        Y.loc[Y.index[0], "country"] = "NEW_COUNTRY"
        model = make_pipeline(X)
        model.fit(X, train[TARGET])
        p = model.predict_proba(Y)[:, 1]
        self.assertEqual(len(p), len(val))
        self.assertTrue(np.isfinite(p).all())

    def test_oversampling_inside_pipeline(self):
        obj = make_pipeline(feature_frame(self.df), "oversample")
        self.assertEqual(list(obj.named_steps), ["prep", "oversample", "classifier"])

    def test_threshold_selection_rejects_unreliable_prediction(self):
        y = np.array([0, 1] * 50)
        p = np.full(len(y), .6)
        table = threshold_table(y, p, min_alerts=10)
        threshold, ok = choose_threshold(table, .95)
        self.assertFalse(ok)
        self.assertGreaterEqual(threshold, .5)

    def test_group_errors_finite(self):
        val = self.df.iloc[:100]
        p = np.linspace(0, 1, 100)
        r = group_errors(val, p, .70)
        self.assertTrue((r["n"] > 0).all())
        self.assertTrue((r["FP"] >= 0).all())

if __name__ == "__main__":
    unittest.main()
