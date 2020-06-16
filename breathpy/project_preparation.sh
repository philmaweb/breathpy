# Creating directories to store results
echo "Starting project setup"
echo "Creating directories"
mkdir -p results/plots/heatmaps
mkdir -p results/plots/clustering
mkdir -p results/plots/roc_curve
mkdir -p results/plots/venn_diagram
mkdir -p results/statistics
mkdir -p results/model
mkdir -p results/data/merged_candy
mkdir -p data/data_tables

# install graphviz for ubuntu
sudo apt-get install graphviz