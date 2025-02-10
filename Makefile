SRC = $(wildcard *.py) $(shell find src scripts -type f -name '*.py')

####### CLIP Smoke Test #######
open_clip_smoke_test:
	python scripts/open_clip_smoke_test.py

profile:
	python -m cProfile -o profile.pstats scripts/open_clip_smoke_test.py
	snakeviz profile.pstats

line_profile:
	kernprof -lv scripts/open_clip_smoke_test.py

####### Formatting #######
format:
	black --line-length 110 $(SRC)
	isort --profile black $(SRC)

####### CLIP Embedding Experiments #######

clip_embed_images_experiment_single_run:
	python scripts/clip_encoding_performance/clip_embed_images_experiment.py --batch_size 128 --function load_each_image_and_encode_immediately

batch_sizes := 1 16 64 256 256 512 640 768 1024
# batch_sizes := 1 2 4 8 16 32 48 64 96 128 160 176 192
# batch_sizes := 224 256 320 384 448 512 576 640 704 768 832 896 960 1024

run_experiment1_load_all_the_images_and_then_encode_them_together:
	for batch_size in $(batch_sizes); do \
		python scripts/clip_encoding_performance/clip_embed_images_experiment.py --batch_size $$batch_size \
			--function load_all_the_images_and_then_encode_them_together \
			--results_csv ./scripts/clip_encoding_performance/clip_embed_images_experiment_MBP_Apple_Model.csv ; \
	done

run_experiment2_load_each_image_and_encode_immediately:
	for batch_size in $(batch_sizes); do \
		python scripts/clip_encoding_performance/clip_embed_images_experiment.py --batch_size $$batch_size \
			--function load_each_image_and_encode_immediately \
			--results_csv ./scripts/clip_encoding_performance/clip_embed_images_experiment_LENNIE_Apple_Model.csv ; \
	done

get_kaggle_data:
	curl -L -o archive.zip \
		https://www.kaggle.com/api/v1/datasets/download/tongpython/cat-and-dog
	unzip -d data archive.zip &> /dev/null

memory_profile:
	python -m memory_profiler scripts/clip_encoding_performance/clip_embed_images_experiment.py

graph:
	python scripts/clip_encoding_performance/graph.py --file_path ./scripts/clip_encoding_performance/clip_embed_images_experiment_LENNIE_Apple_Model.csv

####### Weaviate #######

weaviate-up:
	./weaviate_up.sh

weaviate-script:
	python scripts/weaviate_health/weaviate_single_vector_insertion_test.py


###### CUHK Embeddings ########

untar_cuhk_tar:
	rm -r data/CUHK-PEDES
	mkdir -p data/CUHK-PEDES
	tar -xf CUHK-PEDES.tar.gz -C data/CUHK-PEDES

cuhk_dataset_parquet_to_raw:
	python scripts/cuhk_embeddings/parse_cuhk_pedes.py \
		--raw_folder data/CUHK-PEDES/raw \
		--out_folder data/CUHK-PEDES/out ;

generate_cuhk_embeddings:
	python scripts/cuhk_embeddings/embed_cuhk.py

query_cuhk:
	python scripts/cuhk_embeddings/query_cuhk.py

exclude_one_image_per_set:
	python -m scripts.cuhk_embeddings.exclude_one_image_per_set
