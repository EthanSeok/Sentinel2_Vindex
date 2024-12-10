import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def plot_regression_subplots(group_df, folder, columns_to_compare, output_dir):
    num_cols = len(columns_to_compare)
    rows = (num_cols + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # R² 값과 회귀선 데이터를 저장할 리스트
    regression_results = []

    for i, col in enumerate(columns_to_compare):
        ax = axes[i]

        # x와 y를 바꿔서 플롯 생성
        sns.regplot(
            x=group_df['Mean Value'],
            y=group_df[col],
            ci=None,
            line_kws={'color': 'red'},
            scatter_kws={'s': 80, 'color': 'k'},  # 마커 사이즈 설정
            ax=ax
        )

        # 각 점에 '구역' 컬럼 값을 텍스트로 추가
        for idx, row in group_df.iterrows():
            ax.text(
                row['Mean Value'], row[col],
                str(row['구역']),
                fontsize=11,
                color='black',
                ha='center',
                va='bottom'
            )

        # x와 y를 바꿔서 회귀 분석 수행
        slope, intercept, r_value, p_value, std_err = linregress(group_df['Mean Value'], group_df[col])

        # 회귀선 데이터 계산
        x_vals = np.linspace(group_df['Mean Value'].min(), group_df['Mean Value'].max(), 100)
        y_vals = slope * x_vals + intercept

        # R² 값과 회귀선 데이터를 저장
        regression_results.append({
            'Folder': folder,
            'Column': col,
            'R2': r_value ** 2,
            'Slope': slope,
            'Intercept': intercept,
            'X_Values': x_vals,
            'Y_Values': y_vals
        })

        ax.set_xlabel(f'{folder}')
        ax.set_ylabel(col)

        ax.text(
            0.75, 0.15,
            f"$R^2$: {r_value ** 2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top'
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 그래프 이미지 저장
    image_path = os.path.join(output_dir, f"{folder}.png")
    plt.savefig(image_path)
    plt.close(fig)

    # 회귀선 데이터 저장
    regression_df = pd.DataFrame(regression_results)
    regression_df.to_csv(os.path.join(output_dir, f"{folder}_regression_results.csv"), index=False)


def find_top3_correlations(df, columns_to_compare):
    results = []

    grouped = df.groupby('Folder')
    for folder, group_df in grouped:
        for col in columns_to_compare:
            if len(group_df[col].dropna()) > 1:  # 유효한 데이터가 2개 이상일 때만 수행
                slope, intercept, r_value, p_value, std_err = linregress(group_df['Mean Value'], group_df[col])
                results.append({
                    'Folder': folder,
                    'Column': col,
                    'R2': r_value ** 2
                })

    results_df = pd.DataFrame(results)

    top3_per_column = {}
    for col in columns_to_compare:
        top3 = results_df[results_df['Column'] == col].nlargest(3, 'R2')
        top3_per_column[col] = top3

    return top3_per_column


def main():
    df = pd.read_csv('results.csv')
    df = df.dropna(subset=['요소비료'])

    output_dir = 'output/correlations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    columns_to_compare = ['pH', '유기물', '유효인산', '칼륨', '칼슘', '마그네슘', 'EC', '요소비료']
    grouped = df.groupby('Folder')

    for folder, group_df in grouped:
        plot_regression_subplots(group_df, folder, columns_to_compare, output_dir)

    # 상관관계가 높은 Folder top3 추출
    top3_results = find_top3_correlations(df, columns_to_compare)

    for col, top3 in top3_results.items():
        print(f"\n{col}와 Mean Value의 상관관계(R²)가 가장 높은 Folder Top 3:")
        print(top3)


if __name__ == '__main__':
    main()
