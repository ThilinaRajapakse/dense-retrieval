# Downloads the dataset from https://github.com/facebookresearch/DPR
python download_data.py --resource data.retriever.nq-train
python download_data.py --resource data.retriever.nq-dev

# Move things around and clean up
mv downloads/data .
mv data/retriever/* data
rm -r data/retriever
rm -r downloads
