from . import batched_sequence

def test_argsort_properties():
    for items in (
        tuple(range(10)),
        (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        (0, 2, 4, 6, 8, 1, 3, 5, 7, 9),
        (0, 9, 1, 8, 2, 7, 3, 6, 4, 5),
    ):
        sorted_items, sort_to_orig, orig_to_sort = batched_sequence.argsort(items)

        # orig_to_sort
        for i in range(len(items)):
            assert items[orig_to_sort[i]] == sorted_items[i]
        assert tuple(items[i] for i in orig_to_sort) == sorted_items

        # sort_to_orig
        for i in range(len(items)):
            assert sorted_items[sort_to_orig[i]] == items[i]
        assert tuple(sorted_items[i] for i in sort_to_orig) == items
