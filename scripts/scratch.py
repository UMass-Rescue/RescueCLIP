from collections import Counter

import weaviate
from weaviate.classes.query import Filter

from rescueclip.cuhk import get_sets, get_sets_new
from rescueclip.ml_model import (
    CUHK_Apple_Collection,
    CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection,
    CUHK_ViT_B_32_Collection,
)

in_folder = "/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/out"
stops_file = "/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/caption_all.json"


# Counter on the number of images in each set
def count():
    sets = get_sets_new(in_folder, stops_file)
    print(Counter(len(fls) for fls in sets.values()))


def delete_sets():
    client = weaviate.connect_to_local()
    try:
        sets = get_sets_new(in_folder, stops_file)

        sets_id_to_remove = [set_id for set_id, set_images in sets.items() if len(set_images) < 4]

        assert len(sets_id_to_remove) == 95

        collection = client.collections.get(CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection.name)

        result = collection.data.delete_many(
            where=Filter.by_property("set_number").contains_any(sets_id_to_remove), dry_run=True
        )

        assert result.matches == 200

        result = collection.data.delete_many(
            where=Filter.by_property("set_number").contains_any(sets_id_to_remove),
        )

        assert result.successful == 200
        print(result)
    finally:
        client.close()
