import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style to 'ggplot'
plt.style.use('ggplot')

def plot_freq_classes(df:pd.DataFrame, 
                      class_column:str,
                      plot_no_classes:int=None,
                      bar_color:str='#1f77b4'):
    
    fig = plt.subplots(figsize=(5, plot_no_classes // 3))
    
    # Sort classes by frequency, select the first N and then reverse the selection to plot correctly
    if plot_no_classes is not None:
        top_N_classes = df[class_column]\
                            .value_counts(ascending=False)\
                            .iloc[:N_classes]\
                            .iloc[::-1]
    else:
        top_N_classes = df[class_column]\
                            .value_counts(ascending=False)

    # Plot the horizontal bar chart for the top 15 classes
    ax = top_N_classes.plot.barh(color=bar_color)

    plt.title("Frequency of classes")
    plt.ylabel("")
    sns.despine()

    # Set the xlim to create space for the text
    ax.set_xlim([0, max(top_N_classes)*1.2])  # Increase the x-axis limit by 10%

    for bar in ax.patches:
        # Calculate the space between the bar and the number
        space = bar.get_width() * 0.005  # Adjust this value to increase or decrease the space

        # Use the bar's attributes to place a text label with a space
        plt.text(bar.get_width() + space,  # Add space to the x-coordinate position of text
                bar.get_y() + bar.get_height() / 2,  # y-coordinate position of text
                f'{bar.get_width()}',  # Text to be displayed, bar.get_width() gives the label's value
                va='center')  # Center alignment for the text
    plt.show()
    