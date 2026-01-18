# TensorGuard Integrations Ecosystem

TensorGuardFlow is designed to sit at the center of your MLOps and Robotics stack.

## 1. Robotics Middleware

### ROS 2 (Humble/Iron)
- **Type**: Native `rclpy` Node.
- **Function**: Subscribes to `/camera/image_raw` and `/joint_states`. Publishes encrypted updates to `/tensorguard/update`.
- **Setup**:
    ```bash
    ros2 run tensorguard_ros bridge
    ```

### Formant.io
- **Type**: Cloud Observability Connector.
- **Function**: Visualizes the **System Health** and **Trust Score** directly in your Formant dashboard.
- **Config**: Add the "TensorGuard Plugin" from the Formant catalog.

## 2. Simulation Environments

### NVIDIA Isaac Lab (Omniverse)
- **Type**: Sim-to-Real Connector.
- **Function**:
    - **Data Gen**: Stream synthetic replicator data directly to `DataConnector`.
    - **Validation**: Run the **Evaluation Gate** inside Isaac Sim to verify physics compliance before deploying to real robots.
- **VLA Studio**: Use the "Sim-to-Real" wizard in PEFT Studio to auto-configure domain randomization.

## 3. Deployment Targets

### NVIDIA Jetson (Edge)
- **Optimization**: Native TensorRT integration.
- **Protection**: TrustZone TEE support (Orin Series).

### Kubernetes (Cloud)
- **Chart**: Helm charts available for the Aggregation Server.
- **Scaling**: Horizontal Pod Autoscaling (HPA) based on CPU and Bandwidth metrics.
