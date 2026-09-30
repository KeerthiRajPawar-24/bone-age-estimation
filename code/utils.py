"""
Utility functions for Bone Age Estimation Project.

Author: Keerthi P R
Registration: 252BDA04
Institution: St Joseph's University, Bengaluru
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============ CONFIGURATION ============
IMG_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42


# ============ MODEL BUILDER ============
def build_bone_age_model(img_size=IMG_SIZE, learning_rate=0.001):
    """
    Build the bone age regression model using ResNet50 transfer learning.

    Args:
        img_size: Input image size (default 224 for ResNet50)
        learning_rate: Initial learning rate for Adam optimizer

    Returns:
        Compiled Keras model ready for training
    """
    # Frozen ResNet50 base
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    # Regression head
    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)  # Regression output

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_absolute_error',
        metrics=['mae', 'mse']
    )
    return model


# ============ DATA GENERATORS ============
def create_generators(train_df, val_df, test_df, img_dir,
                      img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create training, validation, and test data generators.

    Training generator uses augmentation; val/test do not.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=img_dir,
        x_col='filename', y_col='boneage',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='raw',
        color_mode='rgb', shuffle=True, seed=RANDOM_SEED
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df, directory=img_dir,
        x_col='filename', y_col='boneage',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='raw',
        color_mode='rgb', shuffle=False
    )
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df, directory=img_dir,
        x_col='filename', y_col='boneage',
        target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='raw',
        color_mode='rgb', shuffle=False
    )
    return train_gen, val_gen, test_gen


# ============ EVALUATION METRICS ============
def compute_metrics(predictions, actuals):
    """
    Compute comprehensive regression metrics.

    Returns dict with MAE, RMSE, R², Pearson r, mean/std residuals.
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    correlation = np.corrcoef(predictions, actuals)[0, 1]
    residuals = predictions - actuals

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r_squared,
        'Pearson_r': correlation,
        'Mean_Residual': np.mean(residuals),
        'Std_Residuals': np.std(residuals),
    }


def bland_altman_analysis(predictions, actuals):
    """
    Compute Bland-Altman metrics for method agreement analysis.

    Returns dict with bias, LoA, and coverage statistics.
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    mean_measurements = (predictions + actuals) / 2
    difference = predictions - actuals

    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    lower_loa = mean_diff - 1.96 * std_diff
    upper_loa = mean_diff + 1.96 * std_diff

    within_loa = np.sum(
        (difference >= lower_loa) & (difference <= upper_loa)
    ) / len(difference) * 100

    return {
        'bias': mean_diff,
        'std_diff': std_diff,
        'lower_loa': lower_loa,
        'upper_loa': upper_loa,
        'coverage_pct': within_loa,
        'mean_measurements': mean_measurements,
        'difference': difference,
    }


def age_stratified_analysis(predictions, actuals):
    """
    Break down errors by age group to identify performance patterns.

    Returns DataFrame with count, MAE, RMSE, mean residual per age bucket.
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    age_bins = [0, 24, 60, 108, 156, 204, 228]
    age_labels = ['0-2y', '2-5y', '5-9y', '9-13y', '13-17y', '17-19y']

    df = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions,
        'abs_error': np.abs(predictions - actuals),
        'residual': predictions - actuals,
    })
    df['age_group'] = pd.cut(df['actual'], bins=age_bins, labels=age_labels)

    stats = df.groupby('age_group', observed=True).agg(
        count=('actual', 'count'),
        mae=('abs_error', 'mean'),
        rmse=('residual', lambda x: np.sqrt(np.mean(x**2))),
        mean_residual=('residual', 'mean'),
    ).round(2)

    return stats


# ============ PLOTTING ============
def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training vs validation loss and MSE over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['loss'], label='Training MAE', color='steelblue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation MAE', color='darkorange', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (months)')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['mse'], label='Training MSE', color='steelblue', linewidth=2)
    axes[1].plot(history['val_mse'], label='Validation MSE', color='darkorange', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Training vs Validation MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actuals(predictions, actuals, r_squared,
                                 save_path='predictions_vs_actuals.png'):
    """Plot scatter of predictions vs actuals with residual plot."""
    residuals = predictions - actuals
    mean_residual = np.mean(residuals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(actuals, predictions, alpha=0.4, color='steelblue', s=15)
    axes[0].plot([0, 250], [0, 250], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Bone Age (months)')
    axes[0].set_ylabel('Predicted Bone Age (months)')
    axes[0].set_title(f'Predicted vs Actual (R² = {r_squared:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(actuals, residuals, alpha=0.4, color='darkorange', s=15)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].axhline(y=mean_residual, color='green', linestyle=':', linewidth=2,
                    label=f'Mean: {mean_residual:.2f}')
    axes[1].set_xlabel('Actual Bone Age (months)')
    axes[1].set_ylabel('Residual (Predicted − Actual)')
    axes[1].set_title('Residual Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bland_altman(ba_results, save_path='bland_altman.png'):
    """Plot Bland-Altman method agreement analysis."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(ba_results['mean_measurements'], ba_results['difference'],
               alpha=0.4, color='steelblue', s=15)
    ax.axhline(y=ba_results['bias'], color='red', linestyle='-', linewidth=2,
               label=f"Bias: {ba_results['bias']:.2f}")
    ax.axhline(y=ba_results['upper_loa'], color='green', linestyle='--', linewidth=2,
               label=f"Upper 95% LoA: {ba_results['upper_loa']:.2f}")
    ax.axhline(y=ba_results['lower_loa'], color='green', linestyle='--', linewidth=2,
               label=f"Lower 95% LoA: {ba_results['lower_loa']:.2f}")
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Mean of Predicted and Actual Age (months)')
    ax.set_ylabel('Predicted − Actual (months)')
    ax.set_title('Bland-Altman Plot: Model Predictions vs Radiologist Labels')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
