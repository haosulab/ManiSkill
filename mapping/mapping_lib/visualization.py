import torch
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def visualize_decoded_features_pca(static_map, implicit_decoder, device="cuda"):
    """
    Visualizes decoded voxel features from a VoxelHashTable using PCA with Open3D.
    The point cloud is colored based on the 3 principal components of the decoded features.
    """
    print("Gathering coordinates from the voxel map...")
    all_coords = []
    with torch.no_grad():
        for level in static_map.levels:
            if hasattr(level, 'coords'):
                all_coords.append(level.coords)
    
    if not all_coords:
        print("No coordinates found in the map to visualize.")
        return
        
    coords_torch = torch.cat(all_coords, dim=0).to(device)
    
    print(f"Visualizing {len(coords_torch)} points from the map.")

    with torch.no_grad():
        # Query features from the map
        feats = static_map.query_voxel_feature(coords_torch)  # [N, feature_dim]
        coords_np = coords_torch.detach().cpu().numpy()

        # Get decoded features
        print("Visualizing decoder features...")
        decoded_feats = implicit_decoder(feats, coords_torch)
        decoded_feats_np = decoded_feats.detach().cpu().numpy()

        # PCA on decoder features
        print("Running PCA on decoded features...")
        pca_decoder = PCA(n_components=3)
        decoded_feats_pca = pca_decoder.fit_transform(decoded_feats_np)
        print("\nDecoder Features PCA explained variance ratio:", pca_decoder.explained_variance_ratio_)
        print("Sum of explained variance ratio:", pca_decoder.explained_variance_ratio_.sum())

        # Normalize for color
        scaler = MinMaxScaler()
        decoded_feats_pca_norm = scaler.fit_transform(decoded_feats_pca)

        # Create PointCloud for visualization
        pcd_decoder = o3d.geometry.PointCloud()
        pcd_decoder.points = o3d.utility.Vector3dVector(coords_np)
        pcd_decoder.colors = o3d.utility.Vector3dVector(decoded_feats_pca_norm)

        # Create a coordinate frame.
        # X-axis: Red, Y-axis: Green, Z-axis: Blue
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # Visualize
        o3d.visualization.draw_geometries([pcd_decoder, coord_frame], window_name="Decoder Features PCA Visualization")
