# visualize_graph.py - Single file untuk visualisasi GNN Graph
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor

def visualize_dengue_gnn():
    """Visualize complete Dengue GNN graph structure"""
    
    print("Analyzing Dengue GNN Graph...")
    
    # Setup
    config = Config()
    preprocessor = DengueDataPreprocessor(config)
    graph_constructor = GraphConstructor(config)
    
    # Load data
    df = preprocessor.load_data("data/test2.csv")
    features, targets, metadata = preprocessor.preprocess_data(df)
    location_coords = metadata['location_coords']
    node_ids = metadata['node_ids']
    
    # Build adjacency matrix
    adj_matrix = graph_constructor.build_spatial_adjacency(location_coords)
    
    print(f"SUCCESS: Graph with {len(node_ids)} nodes, density: {np.mean(adj_matrix > 0):.3f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dengue STGNN - Graph Structure Analysis', fontsize=16, fontweight='bold')
    
    # 1. Adjacency Matrix Heatmap
    ax1 = axes[0, 0]
    short_labels = [id.replace('PKM. ', '').replace(' ', '\n') for id in node_ids]
    sns.heatmap(adj_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=short_labels,
                yticklabels=short_labels,
                cmap='YlOrRd',
                ax=ax1,
                cbar_kws={'label': 'Connection\nStrength'})
    ax1.set_title('Spatial Adjacency Matrix')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0, labelsize=8)
    
    # 2. Network Graph
    ax2 = axes[0, 1]
    G = nx.Graph()
    
    # Add nodes
    for i, node_id in enumerate(node_ids):
        G.add_node(i, label=node_id.replace('PKM. ', ''),
                   pos=(location_coords[i, 1], location_coords[i, 0]))
    
    # Add edges (only significant connections)
    threshold = 0.01
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if adj_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=adj_matrix[i, j])
    
    # Draw network
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                          node_color='lightblue', 
                          node_size=800,
                          edgecolors='black',
                          alpha=0.8)
    
    # Edges with varying thickness
    if weights:
        max_weight = max(weights)
        edge_widths = [w / max_weight * 3 + 0.5 for w in weights]
        nx.draw_networkx_edges(G, pos, ax=ax2,
                              width=edge_widths,
                              alpha=0.6,
                              edge_color='red')
    
    # Labels
    labels = {i: node_id.replace('PKM. ', '').split()[0] for i, node_id in enumerate(node_ids)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=8, font_weight='bold')
    
    ax2.set_title('Network Structure')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Geographic Map with Connections
    ax3 = axes[1, 0]
    
    # Plot puskesmas locations
    scatter = ax3.scatter(location_coords[:, 1], location_coords[:, 0], 
                         s=300, c='red', alpha=0.8, edgecolors='black', linewidth=2)
    
    # Draw connections
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if adj_matrix[i, j] > threshold:
                ax3.plot([location_coords[i, 1], location_coords[j, 1]], 
                        [location_coords[i, 0], location_coords[j, 0]], 
                        'b-', alpha=adj_matrix[i, j] * 2, 
                        linewidth=adj_matrix[i, j] * 5 + 0.5)
    
    # Add labels
    for i, node_id in enumerate(node_ids):
        ax3.annotate(node_id.replace('PKM. ', '').split()[0], 
                    (location_coords[i, 1], location_coords[i, 0]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Geographic Distribution with Connections')
    ax3.grid(True, alpha=0.3)
    
    # 4. Graph Statistics
    ax4 = axes[1, 1]
    
    # Connection strength distribution
    connections = adj_matrix[adj_matrix > 0]
    ax4.hist(connections, bins=12, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(connections), color='red', linestyle='--', 
               label=f'Mean: {np.mean(connections):.3f}')
    ax4.set_xlabel('Connection Strength')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Connection Strength Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Graph Statistics:
Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}  
Density: {nx.density(G):.3f}
Avg Clustering: {nx.average_clustering(G):.3f}
Max Connection: {np.max(adj_matrix):.3f}
Min Connection: {np.min(adj_matrix[adj_matrix > 0]):.3f}"""
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nGraph Summary:")
    print(f"Nodes (Puskesmas): {len(node_ids)}")
    for i, node_id in enumerate(node_ids):
        connections = np.sum(adj_matrix[i] > threshold)
        print(f"  {node_id}: {connections} connections")
    
    print(f"\nAdjacency Matrix:")
    print(f"  Shape: {adj_matrix.shape}")
    print(f"  Non-zero connections: {np.sum(adj_matrix > 0)}")
    print(f"  Average connection strength: {np.mean(connections):.4f}")
    
    return adj_matrix, node_ids, location_coords, G

if __name__ == "__main__":
    try:
        adj_matrix, node_ids, coords, graph = visualize_dengue_gnn()
        print("\nSUCCESS: Graph visualization completed!")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure all required files exist and dependencies are installed.")