import os
import pandas as pd
import yaml
import json
# Import IDTCalculator
from src.idt_calculator_nonreac import IDTCalculator, integrate_idt
# Import plotting utilities
from src.Constrain_plot_utils import ConstraintEfficiencyPlotter

class DataConstraintAnalyzer:
    def __init__(self, config_file):
        # Read configuration file
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set default value derivation logic
        self._set_default_configs()
        
        # Read optimized parameters
        self.optimized_params = self._read_optimized_params()
        
        # Initialize IDTCalculator
        self.idt_calculator = IDTCalculator(
            A1=self.optimized_params['A1'],
            n1=self.optimized_params['n1'],
            E1=self.optimized_params['E1'],
            Ah=self.optimized_params['Ah'],
            nh=self.optimized_params['nh'],
            Eh=self.optimized_params['Eh'],
            Teq=self.optimized_params['Teq'],
            k=self.optimized_params['k'],
            w=self.optimized_params['w'],
            C0=self.optimized_params['C0'],
            xf=self.optimized_params['xf']
        )
        
        # Read JSON configuration
        self.json_config = self._read_json_config()
        
        # Parse CSV file information
        self.csv_files_info = self._parse_csv_files_info()
        
        # Temperature and pressure grouping ranges
        self.temp_range = self.config['temp_range']
        self.pressure_range = self.config['pressure_range']
        self.t_com = self.json_config['timing_parameters']['t_com']
        
        # Constraint efficiency data
        self.constraint_efficiency = {}
    
    def _set_default_configs(self):
        """Set default values for configuration file"""
        results_folder = self.config['results_folder']
        
        # Extract base name from results_folder (remove "_results")
        base_name = results_folder.replace('_results', '')
        
        # Set full path for optimized_params_file
        self.config['optimized_params_file'] = f"./{results_folder}/optimized_parameters.csv"
        
        # Set default value for json_config_file
        if not self.config.get('json_config_file') or self.config['json_config_file'].strip() == '':
            self.config['json_config_file'] = f"{base_name}.json"
        
        # Set default value for output_dir
        if not self.config.get('output_dir') or self.config['output_dir'].strip() == '':
            self.config['output_dir'] = f"./{base_name}_Constraint"
        
        # Set default value for idt_data_file
        if not self.config.get('idt_data_file') or self.config['idt_data_file'].strip() == '':
            self.config['idt_data_file'] = f"{base_name}_IDT.csv"
    
    def _read_optimized_params(self):
        """Read optimized parameters file"""
        params_df = pd.read_csv(self.config['optimized_params_file'])
        params_dict = dict(zip(params_df['Parameter'], params_df['Value']))
        return params_dict
    
    def _read_json_config(self):
        """Read JSON configuration file"""
        with open(self.config['json_config_file'], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_csv_files_info(self):
        """Parse all CSV file information"""
        csv_files_info = []
        
        # Process two_stage section
        if 'two_stage' in self.json_config['file_specifications']:
            two_stage_files = self.json_config['file_specifications']['two_stage']['files']
            if not two_stage_files:
                print("Warning: two_stage files list is empty, skipping...")
            else:
                for file_info in two_stage_files:
                    csv_files_info.append({
                        'path': os.path.join(
                            os.path.dirname(self.config['json_config_file']),
                            self.json_config['file_specifications']['two_stage']['directory'],
                            file_info['name']
                        ),
                        'idt': file_info.get('idt_total', file_info.get('idt_one', file_info.get('idt_ntc'))),
                        'idt_1st': file_info['idt_1st'],
                        'stage': 'two_stage'
                    })
        
        # 处理single_stage部分
        if 'single_stage' in self.json_config['file_specifications']:
            single_stage_files = self.json_config['file_specifications']['single_stage']['files']
            if not single_stage_files:
                print("Warning: single_stage files list is empty, skipping...")
            else:
                for file_info in single_stage_files:
                    csv_files_info.append({
                        'path': os.path.join(
                            os.path.dirname(self.config['json_config_file']),
                            self.json_config['file_specifications']['single_stage']['directory'],
                            file_info['name']
                        ),
                        'idt': file_info.get('idt_total', file_info.get('idt_one', file_info.get('idt_ntc'))),
                        'idt_1st': None,
                        'stage': 'single_stage'
                    })
        
        # 处理ntc_region部分
        if 'ntc_region' in self.json_config['file_specifications']:
            ntc_files = self.json_config['file_specifications']['ntc_region']['files']
            if not ntc_files:
                print("Warning: ntc_region files list is empty, skipping...")
            else:
                print(f"Processing {len(ntc_files)} ntc_region files...")
                for file_info in ntc_files:
                    csv_files_info.append({
                        'path': os.path.join(
                            os.path.dirname(self.config['json_config_file']),
                            self.json_config['file_specifications']['ntc_region']['directory'],
                            file_info['name']
                        ),
                        'idt': file_info['idt_ntc'],
                        'idt_1st': None,
                        'stage': 'ntc_region'
                    })
        
        if not csv_files_info:
            print("Warning: No CSV files found in configuration!")
        
        return csv_files_info
    
    def _group_temp_pressure(self, temp, pressure):
        """将温度和压力分组到指定范围内"""
        temp_group = round(temp / self.temp_range) * self.temp_range
        pressure_group = round(pressure / self.pressure_range) * self.pressure_range
        return temp_group, pressure_group
    
    def _calculate_integral_range(self, df, ig_timing, ig_1st_timing=None):
        """计算一个csv文件中的每时刻温度-压力对应的约束效能（积分区间）"""
        
        # 最终结果始终基于 ig_timing 筛选的数据
        df_result = df[df['Time (msec)'] <= ig_timing].copy()
        
        if len(df_result) < 2:
            return []
        
        # 计算对应的着火进程积分曲线
        # 1. 计算每个时刻的IDT
        df_result['idt_calculated'] = df_result.apply(lambda row: self.idt_calculator.calculate_idt(row['Pressure (bar)'], row['Temperature (K)']), axis=1)
        
        # 2. 使用integrate_idt函数计算积分值
        time_array = df_result['Time (msec)'].values
        idt_array = df_result['idt_calculated'].values
        
        # 计算积分
        integral_total = integrate_idt(time_array, idt_array)
        
        # 将积分值添加到DataFrame
        df_result['integral_total'] = integral_total
        
        # 计算每个时刻（温度,压力)对应的积分区间[上一时刻的积分值,当前时刻的积分值]
        # 注意：第一个时刻的积分区间为[0,当前时刻的积分值]
        df_result['integral_range_total'] = df_result['integral_total'].shift(fill_value=0)
        df_result['integral_range_total'] = df_result[['integral_range_total', 'integral_total']].values.tolist()
        
        # 如果ig_1st_timing存在，则计算integral_range_1st
        if ig_1st_timing is not None:
            # 筛选ig_1st_timing范围内的数据用于计算integral_1st
            df_1st = df[df['Time (msec)'] <= ig_1st_timing].copy()
            
            if len(df_1st) >= 2:
                # 计算ig_1st_timing范围内的IDT_1st值
                df_1st['idt_1st_calculated'] = df_1st.apply(lambda row: self.idt_calculator.calculate_idt_1st(row['Pressure (bar)'], row['Temperature (K)']), axis=1)
                
                # 使用integrate_idt函数计算积分值
                time_array_1st = df_1st['Time (msec)'].values
                idt_1st_array = df_1st['idt_1st_calculated'].values
                
                # 计算积分
                integral_1st = integrate_idt(time_array_1st, idt_1st_array)
                
                # 将积分值添加到df_1st
                df_1st['integral_1st'] = integral_1st
                
                # 计算ig_1st_timing范围内的积分区间
                df_1st['integral_range_1st'] = df_1st['integral_1st'].shift(fill_value=0)
                df_1st['integral_range_1st'] = df_1st[['integral_range_1st', 'integral_1st']].values.tolist()
                
                # 将ig_1st_timing范围内的integral_range_1st信息合并到df_result（基于ig_timing筛选的结果）
                # 只保留在ig_1st_timing范围内的df_1st数据，其他设置为None
                df_1st_relevant = df_1st[['Time (msec)', 'integral_range_1st']].copy()
                
                # 将df_1st_relevant的数据合并到df_result
                df_result = df_result.merge(df_1st_relevant, on='Time (msec)', how='left')
                
                # 对于不在ig_1st_timing范围内的点（即在ig_timing范围内但不在ig_1st_timing范围内的点），integral_range_1st设为None
                df_result['integral_range_1st'] = df_result['integral_range_1st'].where(
                    df_result['Time (msec)'] <= ig_1st_timing, None
                )
                
                
            else:
                # 如果ig_1st_timing的数据不足，所有integral_range_1st设为None
                df_result['integral_range_1st'] = [None] * len(df_result)                
                
        else:
            # 不计算integral_1st相关的内容
            # 返回不包含integral_range_1st的结果
            df_result['integral_range_1st'] = [None] * len(df_result)
            
        results = df_result[['Time (msec)', 'Temperature (K)', 'Pressure (bar)', 'integral_range_total', 'integral_range_1st']].values.tolist()
        return results
    
    def _calculate_idt_values(self, temp, pressure):
        """计算给定温度和压力下的IDT值"""
        idt_total = self.idt_calculator.calculate_idt(pressure, temp)
        idt_1st = self.idt_calculator.calculate_idt_1st(pressure, temp)
        return idt_total, idt_1st
    
    def analyze_constraint_efficiency(self):
        """分析约束效能"""
        if not self.csv_files_info:
            print("No files to process. Skipping analysis.")
            return
        
        all_results = []
        
        for file_info in self.csv_files_info:
            print(f"Processing file: {os.path.basename(file_info['path'])}")
            
            # 读取CSV文件
            try:
                df = pd.read_csv(file_info['path'])
                # 去除列名前后的空格
                df.columns = df.columns.str.strip()
            except Exception as e:
                print(f"Error reading {file_info['path']}: {e}")
                continue
            
            # total_timing 和 first_timing 为IDT + t_com
            total_timing = file_info['idt'] + self.t_com
            if file_info['idt_1st'] is not None:
                first_timing = file_info['idt_1st'] + self.t_com
            else:
                first_timing = None

            # 使用_calculate_integral_range获得返回结果
            results = self._calculate_integral_range(df, total_timing, first_timing)
            
            # 添加当前文件的结果到总结果列表
            all_results.extend(results)
        
        # 将结果转换为DataFrame，所有结果都应包含5列（包括integral_range_1st，可能为None）
        df_results = pd.DataFrame(all_results, columns=['Time (msec)', 'Temperature (K)', 'Pressure (bar)', 'integral_range_total', 'integral_range_1st'])
        
        # 对温度和压力数据进行范围更新（将一定范围内的温度和压力值统一为一个值）
        # 使用配置中定义的范围进行分组
        df_results['Temperature (K)'], df_results['Pressure (bar)'] = zip(*df_results.apply(
            lambda row: self._group_temp_pressure(row['Temperature (K)'], row['Pressure (bar)']), axis=1
        ))
        
        # 定义函数来合并积分区间（求并集）
        def merge_integral_ranges(ranges):
            # 检查ranges是否为Series或列表，如果是Series则转换为列表
            if hasattr(ranges, 'iloc'):  # 如果是pandas Series
                range_list = ranges.dropna().tolist()
            else:  # 如果是列表或其他类型
                if ranges is None:
                    return []
                range_list = [ranges] if not hasattr(ranges, '__iter__') or isinstance(ranges, str) else list(ranges)
            
            # 过滤掉None值和无效区间
            valid_ranges = []
            for r in range_list:
                if r is not None and hasattr(r, '__iter__') and not isinstance(r, str):
                    try:
                        r_list = list(r)
                        if len(r_list) == 2:
                            valid_ranges.append(r_list)
                    except (TypeError, ValueError):
                        continue
            
            if not valid_ranges:
                return []
            
            # 按照区间的起始点排序
            valid_ranges.sort(key=lambda x: x[0])
            
            # 合并重叠的区间
            merged = [valid_ranges[0]]
            for current in valid_ranges[1:]:
                last = merged[-1]
                # 如果当前区间与上一个区间重叠或相邻，则合并
                if current[0] <= last[1]:
                    merged[-1] = [last[0], max(last[1], current[1])]
                else:
                    merged.append(current)
            
            return merged
        
        # 对每个（温度，压力）组合进行分组，并合并积分区间
        grouped = df_results.groupby(['Temperature (K)', 'Pressure (bar)'])
        
        # 为每个组计算合并后的积分区间
        merged_total_ranges = grouped['integral_range_total'].apply(merge_integral_ranges)
        merged_1st_ranges = grouped['integral_range_1st'].apply(merge_integral_ranges)
        
        # 为每个组计算合并后积分区间的长度
        def calculate_range_length(ranges):
            if not ranges:
                return 0
            total_length = 0
            for r in ranges:
                total_length += r[1] - r[0]  # 区间长度 = 上界 - 下界
            return total_length
        
        merged_total_lengths = merged_total_ranges.apply(calculate_range_length)
        merged_1st_lengths = merged_1st_ranges.apply(calculate_range_length)
        
        # 创建包含合并后结果的DataFrame（只包含最终需要的数据）
        result_summary = pd.DataFrame({
            'Temperature (K)': merged_total_ranges.index.get_level_values('Temperature (K)'),
            'Pressure (bar)': merged_total_ranges.index.get_level_values('Pressure (bar)'),
            'merged_integral_total_length': merged_total_lengths.values,
            'merged_integral_1st_length': merged_1st_lengths.values
        })
        
        # 将汇总结果转换为绘图函数期望的字典格式
        # 每个（温度，压力）组合对应的积分区间长度就是该组合的约束效能
        self.constraint_efficiency = {}
        for idx, row in result_summary.iterrows():
            # 使用温度和压力作为键
            key = (row['Temperature (K)'], row['Pressure (bar)'])
            # 将积分区间长度作为约束效能值
            self.constraint_efficiency[key] = {
                'temp': row['Temperature (K)'],
                'pressure': row['Pressure (bar)'],
                'efficiency_total': row['merged_integral_total_length'],
                'efficiency_1st': row['merged_integral_1st_length'] if pd.notna(row['merged_integral_1st_length']) else 0
            }  
   
    def run(self):
        """运行完整分析流程"""
        print("Starting constraint efficiency analysis...")
        self.analyze_constraint_efficiency()
        
        # 使用绘图工具类进行可视化和数据保存
        plotter = ConstraintEfficiencyPlotter(self.config)
        plotter.plot_3d_constraint_efficiency(self.constraint_efficiency)
        plotter.plot_2d_constraint_efficiency(self.constraint_efficiency)
        plotter.save_constraint_efficiency_data(self.constraint_efficiency)
        print("Analysis completed!")

if __name__ == "__main__":
    # 使用配置文件初始化分析器
    analyzer = DataConstraintAnalyzer('ConstraintAnalysis_config.yaml')
    # 运行分析
    analyzer.run()
