import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style to 'ggplot'
plt.style.use('ggplot')



def plot_freq_classes(df:pd.DataFrame, 
                      class_column:str,
                      plot_no_classes:int=None,
                      bar_color:str='#1f77b4'):
    """Create a horizontal bar plot of frequency classes.

    Args:
        df (pd.DataFrame): Dataframe containing the class.
        class_column (str): Name of the column in df that contains the class label.
        plot_no_classes (int, optional): Number of classes to plot.
        bar_color (str, optional): Color of the bars as HEX value.
    """
    fig = plt.subplots(figsize=(5, plot_no_classes // 3))
    
    # Sort classes by frequency, select the first N and then reverse the selection to plot correctly
    if plot_no_classes is not None:
        top_N_classes = df[class_column]\
                            .value_counts(ascending=False)\
                            .iloc[:plot_no_classes]\
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



def plot_tokens_per_class(df:pd.DataFrame, 
                          class_column:str,
                          tokens_cnt_column:str
                          ):
    """Plot a box-plot of the number of tokens per sequence. All classes are
    plotted in a decreasing order given by the median value.

    Args:
        df (pd.DataFrame): Dataframe containing the class_column and tokens_cnt_column.
        class_column (str): Name of the column in df that contains the class label.
        tokens_cnt_column (str): Name of the column in df that contains the number of tokens per sequence.
    """
    fig = plt.subplots(figsize=(5,20))

    # Calculate the medians and sort the labels
    medians = df.groupby(class_column)[tokens_cnt_column]\
                .median()\
                .sort_values(ascending=False)

    # The index of medians now holds the label_names in the order you want
    sorted_labels = medians.index.tolist()

    # Create a horizontal boxplot
    sns.boxplot(x=tokens_cnt_column, y=class_column, 
                data=df, 
                orient='h',
                order=sorted_labels, 
                showfliers=True)

    sns.despine()
    plt.show()