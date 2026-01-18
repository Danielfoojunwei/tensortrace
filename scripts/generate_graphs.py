
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

def draw_fl_graph(output_path):
    """Draws a Hub-and-Spoke Federated Learning architecture."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Define positions
    center = (0.5, 0.5)
    spokes = [
        (0.2, 0.8), (0.8, 0.8),
        (0.1, 0.5), (0.9, 0.5),
        (0.2, 0.2), (0.8, 0.2)
    ]
    
    # Draw connections
    for spoke in spokes:
        plt.plot([center[0], spoke[0]], [center[1], spoke[1]], 
                 color='#0ea5e9', linestyle='--', alpha=0.5, zorder=1)
        
        # Draw gradient flow arrows (simplified)
        mid_x = (center[0] + spoke[0]) / 2
        mid_y = (center[1] + spoke[1]) / 2
        plt.arrow(spoke[0], spoke[1], (center[0]-spoke[0])*0.4, (center[1]-spoke[1])*0.4,
                  head_width=0.02, color='#10b981', alpha=0.8, zorder=2)

    # Draw Central Server
    circle = plt.Circle(center, 0.12, color='#0f172a', ec='#0ea5e9', lw=3, zorder=3)
    ax.add_patch(circle)
    plt.text(center[0], center[1], "Global\nModel\n(Aggregation)", 
             ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Draw Agents
    for i, spoke in enumerate(spokes):
        rect = patches.FancyBboxPatch((spoke[0]-0.08, spoke[1]-0.06), 0.16, 0.12,
                                      boxstyle="round,pad=0.02",
                                      fc='#1e293b', ec='#f59e0b', lw=2, zorder=3)
        ax.add_patch(rect)
        plt.text(spoke[0], spoke[1], f"Agent {i+1}\n(Train)", 
                 ha='center', va='center', color='white', fontsize=9)

    plt.title("Federated Learning (FL) Architecture\nSecure Aggregation Topology", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def draw_oft_graph(output_path):
    """Draws a conceptual diagram of Orthogonal Finetuning (OFT)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pre-trained Weights (Fixed)
    rect_base = patches.Rectangle((0.1, 0.3), 0.2, 0.4, linewidth=2, edgecolor='#334155', facecolor='#cbd5e1')
    ax.add_patch(rect_base)
    ax.text(0.2, 0.5, "W_0\n(Frozen)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Multiplication Sign
    ax.text(0.35, 0.5, "×", ha='center', va='center', fontsize=24, color='#64748b')
    
    # Orthogonal Matrix R (Block Diagonal)
    rect_ortho = patches.Rectangle((0.4, 0.3), 0.2, 0.4, linewidth=2, edgecolor='#0ea5e9', facecolor='#e0f2fe')
    ax.add_patch(rect_ortho)
    
    # Blocks inside R
    for i in range(4):
        y = 0.31 + i*0.095
        block = patches.Rectangle((0.41, y), 0.18, 0.08, linewidth=1, edgecolor='#0284c7', facecolor='#38bdf8', alpha=0.5)
        ax.add_patch(block)
        
    ax.text(0.5, 0.5, "R\n(Orthogonal)", ha='center', va='center', fontsize=12, fontweight='bold', color='#0369a1')
    
    # Equals Sign
    ax.text(0.65, 0.5, "=", ha='center', va='center', fontsize=24, color='#64748b')
    
    # Resulting Adapted Weights
    rect_res = patches.Rectangle((0.7, 0.3), 0.2, 0.4, linewidth=2, edgecolor='#10b981', facecolor='#d1fae5')
    ax.add_patch(rect_res)
    ax.text(0.8, 0.5, "W'\n(Adapted)", ha='center', va='center', fontsize=12, fontweight='bold', color='#047857')

    # Annotations
    ax.annotate("Preserves Hyperspherical Energy", xy=(0.5, 0.28), xytext=(0.5, 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Orthogonal Finetuning (OFT) Mechanism\nStable Adaptation via Orthogonal Transformation", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating graphs...")
    draw_fl_graph("artifacts/fl_architecture.png")
    draw_oft_graph("artifacts/oft_mechanism.png")
    print("Graphs generated in artifacts/")
