"""filter_reviews dedup must not collide two genuinely different reviews
that happen to share a long templated opening."""
import unittest

from pipeline.models import Review
from pipeline.pdf_parser import filter_reviews


def review(rid, author, date_, body):
    return Review(review_id=rid, product="P", author=author, rating=5.0,
                  title="", date=date_, variant="", body=body)


class TestFilterReviewsDedup(unittest.TestCase):
    def test_distinct_bodies_sharing_a_long_prefix_both_survive(self):
        shared = "This product works great and I would recommend it to anyone " \
                 "who is looking for something reliable and easy to use every day"
        self.assertGreater(len(shared), 120)
        a = review("r1", "Jo", "2026-01-01", shared + " — five stars overall.")
        b = review("r2", "Jo", "2026-01-01", shared + " — but shipping was slow.")
        out = filter_reviews([a, b])
        self.assertEqual(len(out), 2)

    def test_true_duplicate_still_collapses(self):
        a = review("r1", "Jo", "2026-01-01", "Great product, works as expected.")
        b = review("r2", "Jo", "2026-01-01", "Great product, works as expected.")
        out = filter_reviews([a, b])
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
