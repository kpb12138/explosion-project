import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


class ConstraintEfficiencyPlotter:
    """Constraint efficiency plotting and data saving utility class"""
    
    def __init__(self, config):
        self.config = config
    
    def _prepare_data(self, constraint_efficiency):
        """Prepare data: extract temperature, pressure and efficiency values"""
        if not constraint_efficiency:
            return None, None, None
        
        temps = []
        pressures = []
        efficiency_total = []
        efficiency_1st = []
        
        for key, data in constraint_efficiency.items():
            temps.append(data['temp'])
            pressures.append(data['pressure'])
            # Add total efficiency and first-stage efficiency
            efficiency_total.append(data['efficiency_total'])            
            efficiency_1st.append(data['efficiency_1st'])
        
        return temps, pressures, efficiency_total, efficiency_1st
    
    def _create_custom_cmap(self):
        """Create custom colormap: white for minimum, darker colors (brown series) for higher values"""
        # Ensure the first point of colormap strictly corresponds to minimum value (white)
        # Color sequence: white -> light yellow -> orange -> brown -> dark brown
        # Ensure the first color point (0.0) corresponds to white and the last (1.0) corresponds to darkest color
        colors = [(0.0, (1, 1, 1)),   # White, corresponds to vmin
                  (0.2, (1, 0.9, 0.8)),  # Light yellow
                  (0.4, (1, 0.7, 0.5)),  # Orange
                  (0.6, (0.9, 0.5, 0.2)),  # Brown
                  (1.0, (0.8, 0.3, 0.1))]  # Dark brown, corresponds to vmax
        
        cmap = LinearSegmentedColormap.from_list('white_brown', colors, N=100)
        # Set under color to white to ensure any value below vmin shows as white
        cmap.set_under(color='white')
        return cmap
    
    def _filter_and_interpolate_data(self, temps, pressures, efficiencies, grid_size=100, smooth_sigma=1):
        """Data preprocessing: filter data points and perform interpolation and smoothing"""
        # Convert to numpy arrays
        x = np.array(temps)
        y = np.array(pressures)
        z = np.array(efficiencies)
        
        # Set new minimum threshold
        threshold = 1e-6
        
        # Data preprocessing: set values <= 0 to threshold
        z[z <= 0] = threshold        
        # Create grid
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)
        
        # Implement logarithmic interpolation: take log of z, linear interpolate, then exp back
        log_z = np.log(z)
        log_zi = griddata((x, y), log_z, (xi, yi), method='linear')
        zi = np.exp(log_zi)
        
        # Handle NaN values
        zi = np.nan_to_num(zi, nan=threshold * 10.0)
        
        # Ensure data is within reasonable range
        z_max = zi.max()
        
        # Smooth processing
        zi_smooth = gaussian_filter(zi, sigma=smooth_sigma)
        # Ensure smoothed data is not less than threshold
        zi_smooth[zi_smooth <= threshold] = threshold * 10.0
        
        return xi, yi, zi_smooth, z_max
    
    def plot_3d_constraint_efficiency(self, constraint_efficiency):
        """Plot 3D constraint efficiency graph"""
        # Prepare data
        temps, pressures, efficiency_total, efficiency_1st = self._prepare_data(constraint_efficiency)
        
        # Check if data exists
        if not temps:
            print("No constraint efficiency data available. Skipping plot generation.")
            return
        
        # Plot 3D graph for efficiency_total
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use interpolation to create smooth surface
        try:
            # Data preprocessing, interpolation and smoothing
            xi, yi, zi_smooth, z_max = self._filter_and_interpolate_data(
                temps, pressures, efficiency_total, grid_size=100, smooth_sigma=1
            )
            
            # Create custom colormap
            custom_cmap = self._create_custom_cmap()
            
            # Plot smooth surface with logarithmic color mapping, minimum fixed at 1e-5
            surf = ax.plot_surface(
                xi, yi, zi_smooth, 
                cmap=custom_cmap, 
                alpha=0.8, 
                edgecolor='none',
                norm=LogNorm(vmin=1e-5, vmax=1.0)
            )
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax)
            # Adjust colorbar position for 3D plot
            cbar.shrink = 0.5
            cbar.aspect = 10
            cbar.set_label(self.config['z_label'], labelpad=15)
            
        except ImportError:
            # Fallback to scatter plot if scipy not available
            print("Warning: scipy not available, please install scipy.")
            # Exit program
            sys.exit(1)
            
        except Exception as e:
            # Fallback to scatter plot for other errors
            print(f"Warning: Error creating surface plot: {e}, falling back to scatter plot.")
            sys.exit(1)
        
        # Set labels and title
        ax.set_xlabel(self.config['x_label'])
        ax.set_ylabel(self.config['y_label'])
        ax.set_zlabel(self.config['z_label'], labelpad=8)  # Increase padding to avoid overlap
        ax.set_title(self.config['plot_title'])              
        
        # Save efficiency_total plot
        output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_plot_path = os.path.join(output_dir, self.config['output_plot_file'])
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"3D plot for efficiency_total saved to: {output_plot_path}")
        
        plt.close()
        
        # Check if efficiency_1st is non-zero and has more than 3 points
        if efficiency_1st is not None and len(efficiency_1st) > 3 and np.count_nonzero(efficiency_1st) > 0:
            print("Plotting 3D graph for efficiency_1st...")
            
            # Create new figure for efficiency_1st
            fig_1st = plt.figure(figsize=(12, 8))
            ax_1st = fig_1st.add_subplot(111, projection='3d')
            
            try:
                # Apply same logic to plot 3D graph for efficiency_1st
                xi_1st, yi_1st, zi_smooth_1st, z_max_1st = self._filter_and_interpolate_data(
                    temps, pressures, efficiency_1st, grid_size=100, smooth_sigma=1
                )
                
                # Create custom colormap
                custom_cmap_1st = self._create_custom_cmap()
                
                # Plot smooth surface for efficiency_1st
                surf_1st = ax_1st.plot_surface(
                    xi_1st, yi_1st, zi_smooth_1st, 
                    cmap=custom_cmap_1st, 
                    alpha=0.8, 
                    edgecolor='none',
                    norm=LogNorm(vmin=1e-5, vmax=1.0)
                )
                
                # Add colorbar
                cbar_1st = fig_1st.colorbar(surf_1st, ax=ax_1st)
                # Adjust colorbar position for 3D plot
                cbar_1st.shrink = 0.5
                cbar_1st.aspect = 10
                
                # Set z-axis label with '1st' suffix for efficiency_1st
                z_label_1st = self.config['z_label'] + "_1st"
                cbar_1st.set_label(z_label_1st, labelpad=15)
                
            except ImportError:
                # Fallback to scatter plot if scipy not available
                print("Warning: scipy not available, please install scipy.")
                # Exit program
                sys.exit(1)
                
            except Exception as e:
                # Fallback to scatter plot for other errors
                print(f"Warning: Error creating surface plot for efficiency_1st: {e}, falling back to scatter plot.")
                sys.exit(1)
            
            # Set labels and title for efficiency_1st plot
            ax_1st.set_xlabel(self.config['x_label'])
            ax_1st.set_ylabel(self.config['y_label'])
            ax_1st.set_zlabel(z_label_1st, labelpad=8)  # Increase padding to avoid overlap
            ax_1st.set_title(self.config['plot_title'] + " (1st)")
            
            # Save efficiency_1st plot as separate file with '1st' suffix
            output_plot_path_1st = os.path.join(output_dir, self.config['output_plot_file'].replace('.', '_1st.'))
            plt.savefig(output_plot_path_1st, dpi=300, bbox_inches='tight')
            print(f"3D plot for efficiency_1st saved to: {output_plot_path_1st}")
            
            plt.close()
        

    
    def plot_2d_constraint_efficiency(self, constraint_efficiency):
        """Plot 2D contour map showing temperature-pressure-constraint efficiency relationship"""
        # Prepare data
        temps, pressures, efficiency_total, efficiency_1st = self._prepare_data(constraint_efficiency)
        
        # Check if data exists
        if not temps:
            print("No valid data points for 2D plotting. Skipping plot generation.")
            return
        
        # Plot 2D graph for efficiency_total
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Use interpolation to create smooth contours
        try:
            # Data preprocessing, interpolation and smoothing
            xi, yi, zi_smooth, z_max = self._filter_and_interpolate_data(
                temps, pressures, efficiency_total, grid_size=100, smooth_sigma=1
            )
            
            # Create custom colormap
            custom_cmap = self._create_custom_cmap()
            
            # Plot contour map with logarithmic color mapping
            contourf = ax.contourf(
                xi, yi, zi_smooth, 
                cmap=custom_cmap, 
                alpha=0.8, 
                levels=20,
                norm=LogNorm(vmin=1e-5, vmax=1.0)
            )
            
            # Add contours
            ax.contour(xi, yi, zi_smooth, colors='black', linewidths=0.5, levels=10)
            
            # Add colorbar
            cbar = fig.colorbar(contourf, ax=ax)
            cbar.shrink = 0.5
            cbar.aspect = 10
            cbar.set_label(self.config['z_label'], labelpad=15)
            
        except ImportError:
            # Fallback to scatter plot if scipy not available
            print("Warning: scipy not available, please install scipy.")
            exit(1)
            
        except Exception as e:
            # Fallback to scatter plot for other errors
            print(f"Warning: Error creating 2D contour plot: {e}.")
            exit(1)            
        
        # Set labels and title
        ax.set_xlabel(self.config['x_label'])
        ax.set_ylabel(self.config['y_label'])
        ax.set_title(f"2D {self.config['plot_title']}")
        
        # Save efficiency_total plot
        output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_plot_path = os.path.join(output_dir, self.config['output_2d_plot_file'])
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"2D plot for efficiency_total saved to: {output_plot_path}")
        
        plt.close()
        
        # Check if efficiency_1st is non-zero and has more than 3 points
        if efficiency_1st is not None and len(efficiency_1st) > 3 and np.count_nonzero(efficiency_1st) > 0:
            print("Plotting 2D graph for efficiency_1st...")
            
            # Create new figure for efficiency_1st
            fig_1st = plt.figure(figsize=(12, 8))
            ax_1st = fig_1st.add_subplot(111)
            
            try:
                # Apply same logic to plot 2D graph for efficiency_1st
                xi_1st, yi_1st, zi_smooth_1st, z_max_1st = self._filter_and_interpolate_data(
                    temps, pressures, efficiency_1st, grid_size=100, smooth_sigma=1
                )
                
                # Create custom colormap
                custom_cmap_1st = self._create_custom_cmap()
                
                # Plot contour map for efficiency_1st
                contourf_1st = ax_1st.contourf(
                    xi_1st, yi_1st, zi_smooth_1st, 
                    cmap=custom_cmap_1st, 
                    alpha=0.8, 
                    levels=20,
                    norm=LogNorm(vmin=1e-5, vmax=1.0)
                )
                
                # Add contours
                ax_1st.contour(xi_1st, yi_1st, zi_smooth_1st, colors='black', linewidths=0.5, levels=10)
                
                # Add colorbar
                cbar_1st = fig_1st.colorbar(contourf_1st)
                
                # Set z-axis label with '1st' suffix for efficiency_1st
                z_label_1st = self.config['z_label'] + "_1st"
                cbar_1st.set_label(z_label_1st, labelpad=15)
                
            except ImportError:
                # Fallback to scatter plot if scipy not available
                print("Warning: scipy not available, please install scipy.")
                exit(1)
                
            except Exception as e:
                # Fallback to scatter plot for other errors
                print(f"Warning: Error creating 2D contour plot for efficiency_1st: {e}.")
                exit(1)
            
            # Set labels and title for efficiency_1st plot
            ax_1st.set_xlabel(self.config['x_label'])
            ax_1st.set_ylabel(self.config['y_label'])
            ax_1st.set_title(f"2D {self.config['plot_title']} (1st)")
            
            # Save efficiency_1st plot as separate file with '1st' suffix
            output_plot_path_1st = os.path.join(output_dir, self.config['output_2d_plot_file'].replace('.', '_1st.'))
            plt.savefig(output_plot_path_1st, dpi=300, bbox_inches='tight')
            print(f"2D plot for efficiency_1st saved to: {output_plot_path_1st}")
            
            plt.close()
    
    def save_constraint_efficiency_data(self, constraint_efficiency):
        """Save constraint efficiency data"""
        # Check if data exists
        if not constraint_efficiency:
            print("No constraint efficiency data available. Skipping data saving.")
            return
        
        # Prepare data
        data = []
        for key, item in constraint_efficiency.items():
            data.append({
                'Temperature (K)': item['temp'],
                'Pressure (bar)': item['pressure'],
                'Constraint Efficiency (Total)': item['efficiency_total'],
                'Constraint Efficiency (1st)': item['efficiency_1st']
            })
        
        # 检查是否有有效数据
        if not data:
            print("No valid data points to save. Skipping data saving.")
            return
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_data_path = os.path.join(output_dir, self.config['output_data_file'])
        df.to_csv(output_data_path, index=False)
        print(f"Constraint efficiency data saved to: {output_data_path}")