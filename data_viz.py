import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.figure_factory as ff
import os
import plotly.offline
import data_prep
import data_analysis

def output_graph(df: pd.DataFrame, 
                 controls: dict, 
                 lasso: bool=False, 
                 category: str=None,
                 online: bool=False) -> (pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper):
    """
    Runs double selection and plots chart of results coefficients along with unconditional pay gaps and respective confidence intervals
    
    """

    controls_adj = [category+"_"+control for control in controls[category]["controls"] if controls[category]["omitted"] != control]

    if lasso:
        exog = data_analysis.double_selection(df, controls, controls_adj, category).assign(const=1)
    else:
        exog = df.drop("Income", axis=1).assign(const=1)

    est_full = sm.OLS(endog=df["Income"], 
                      exog=exog).fit()
    
    est_simple = sm.OLS(endog=df["Income"], 
                        exog=df.drop("Income", 
                        axis=1)[controls_adj]
                        .assign(const=1)).fit()
    
    output_full = pd.concat([est_full.params, est_full.bse*1.96], 
                            axis=1).rename(columns={0: "coef_full", 1: "conf_95_full"}).filter(like=category+"_", axis=0)

    output_simple = pd.concat([est_simple.params, est_simple.bse*1.96], 
                              axis=1).rename(columns={0: "coef_simple", 1: "conf_95_simple"}).filter(like=category+"_", axis=0)
    
    output = pd.concat([output_full, output_simple], axis=1)

    output.index = [x.replace(category + "_", "") for x in output.index]

    if category in data_prep.num_columns:
        output.sort_index(axis=0, 
                          ascending=True, 
                          inplace=True)
    else:
        output.sort_values(by="coef_full", 
                           ascending=False, 
                           inplace=True)

    if category == "OrgSize":
        output = output.T[['just_me_i_am_a_freelancer_sole_proprietor', '2_9', '10_to_19', '20_to_99', '100_to_499', 
                         '500_to_999', '1_000_to_4_999', '5_000_to_9_999', 'no_answer']].T
    
    output["explained"] = output["coef_simple"] - output["coef_full"]

    for x, y, z in zip(["upper_simple", "upper_full"], 
                       ["coef_simple", "coef_full"], 
                       ["conf_95_simple", "conf_95_full"]): output[x] = data_analysis.unlog(output[y] + output[z])

    for x, y, z in zip(["lower_simple", "lower_full"], 
                       ["coef_simple", "coef_full"], 
                       ["conf_95_simple", "conf_95_full"]): output[x] = data_analysis.unlog(output[y] - output[z])

    output["coef_simple"] = data_analysis.unlog(output["coef_simple"])
    output["coef_full"] = data_analysis.unlog(output["coef_full"])

    for x, y  in zip(["upper_simple", "upper_full"], 
                     ["coef_simple", "coef_full"]): output[x] = output[x] - output[y]

    for x, y  in zip(["lower_simple", "lower_full"], 
                     ["coef_simple", "coef_full"]): output[x] = output[y] - output[x]

    try: 
        output.drop(labels=["no_answer"], inplace=True)
    except:
        pass
    
    output_index = list(output.index)

    my_range=np.array(range(1,len(output_index)+1))

    output_index = [x[:16] for x in output.index]
    
    for i in range(0, len(output_index), 2):
       output_index[i] = "<br>" + output_index[i]

    fig = go.Figure(data=[
        go.Bar(name="Unadjusted", 
               x=my_range, 
               y=output["coef_simple"], 
               marker_color="#d7eaf7", 
               showlegend=False, 
               width=min(0.28, 0.28*len(my_range)/4), 
               hoverinfo="skip"),
        go.Bar(name="Adjusted", 
               x=my_range, 
               y=output["coef_full"], 
               marker_color="#ffcdcd",
               showlegend=False, 
               width=min(0.28, 0.28*len(my_range)/4), 
               hoverinfo="skip"),
        go.Scatter(name="Unadjusted", 
                   x=my_range-0.2, 
                   y=output["coef_simple"], 
                   mode="markers", 
                   marker_color="#3493d3", 
                   marker_size=20, 
                   error_y=dict(type="data", 
                                symmetric=False, 
                                array=output["upper_simple"], 
                                arrayminus=output["lower_simple"], 
                                width=5, 
                                thickness=2, 
                                visible=True),
                   hovertemplate="%{text}",
                   text=[name.replace("<br>", "") + 
                         ": {:+,.1%}".format(coef) + 
                         ", 95% conf: [{:+,.1%}".format(lower) + 
                         ", {:+,.1%}]".format(upper) for name, coef, upper, lower in zip(output_index, 
                                                                                        output["coef_simple"], 
                                                                                        output["coef_simple"] + output["upper_simple"], 
                                                                                        output["coef_simple"] - output["lower_simple"])]),
        go.Scatter(name="Adjusted", 
                   x=my_range+0.2, 
                   y=output["coef_full"], 
                   mode="markers", 
                   marker_color="red", 
                   marker_size=20,
                   error_y=dict(type="data", 
                                symmetric=False, 
                                array=output["upper_full"], 
                                arrayminus=output["lower_full"], 
                                width=5, 
                                thickness=2, 
                                visible=True),
                   hovertemplate="%{text}",
                   text=[name.replace("<br>", "") + 
                         ": {:+,.1%}".format(coef) + 
                         ", 95% conf: [{:+,.1%}".format(lower) + 
                         ", {:+,.1%}]".format(upper) for name, coef, upper, lower in zip(output_index, 
                                                                                        output["coef_full"], 
                                                                                        output["coef_full"] + output["upper_full"], 
                                                                                        output["coef_full"] - output["lower_full"])])],
                   layout=go.Layout(title=go.layout.Title(text="Income vs. " + controls[category]["title"])))

    fig.update_layout(barmode="group", 
                      bargroupgap=0.1,
                      xaxis = dict(tickmode = "array",
                                   tickvals = my_range,
                                   ticktext = output_index,
                                   tickangle = 0,
                                   ticks = "outside",
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black",
                                   range=[0.5, max(my_range)+0.5]),
                      yaxis = dict(tickformat = ".1%",
                                   ticks = "outside",
                                   title = "Pay Gap vs. " + controls[category]["omitted"][:16],
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black",
                                   showgrid=True,
                                   gridcolor="#e8e8e8",
                                   gridwidth=0.5,
                                   zeroline=True,
                                   zerolinecolor="#e8e8e8",
                                   zerolinewidth=2),
                      #width=max(750, len(my_range) * 100),
                      margin = dict(r=0),
                      plot_bgcolor="white",
                      title_font = dict(size=20),
                      legend=go.layout.Legend(x=0,
                                              y=1,
                                              yanchor="bottom",
                                              orientation="h"))

    if not os.path.exists("images"):
        os.mkdir("images")
        
    if not os.path.exists("embeds"):
        os.mkdir("embeds")

    fig.write_image("images/" + category + ".jpg", scale=3)
    #fig.write_image("images/" + category + ".jpg", width=1000, height=500, scale=3)
    fig.write_html("embeds/" + category + ".html", include_plotlyjs='cdn')
    #print(plotly.offline.plot(fig, include_plotlyjs=False, output_type='div'))

    if online:
        py.plot(fig, filename=category)
    
    fig.show()
    
    return output, est_full

def output_waterfall(df: pd.DataFrame, 
                     coef_full: float, 
                     controls: dict, 
                     category: str,
                     exp: str,
                     online: bool=False) -> None:
    """
    Plots waterfall breakdown of explained pay gap 
    
    """

    coef_simple = coef_full + df.values.flatten().sum()

    step_names = ["Adjusted"] + list(df.index) + ["Unadjusted"]

    steps = np.array([coef_full] + list(df.values.flatten()) + [coef_simple])

    new_steps = [0] * len(steps)

    for step in range(len(steps)):
        if step == 0:
            new_steps[step] = data_analysis.unlog(steps[step])
        elif step == len(steps) - 1:
            new_steps[step] = data_analysis.unlog(coef_simple)
        else:
            new_steps[step] = data_analysis.unlog(sum(steps[:step+1])) - sum(new_steps[:step])

    steps = new_steps

    my_range=np.array(list(range(1, len(steps)+1)))

    for i in range(0, len(step_names), 2):
       step_names[i] = "<br>" + step_names[i]

    fig = go.Figure(go.Waterfall(orientation="v",
                                 measure=["absolute"] + (["relative"] * len(df.index)) + ["total"],
                                 x=step_names,
                                 text=["{:,.1%}".format(steps[0])] + 
                                      ["{:+,.1%}".format(step) for step in steps[1:-1]] + 
                                      ["{:,.1%}".format(steps[-1])],
                                 textposition="outside",
                                 y=steps,
                                 connector = {"line":{"color":"grey"}},
                                 hovertemplate="%{text}",
                                 name=df.columns[0],
                                 totals = {"marker":{"color":"#3493d3"}},
                                 decreasing = {"marker":{"color":"#ff4747"}}))

    fig.update_layout(title = "Drivers of the " + df.columns[0][:16] + " vs. " + controls[category]["omitted"][:16] + " pay gap", 
                      xaxis = dict(tickmode = "array",
                                   tickvals = step_names,
                                   tickangle = 0,
                                   ticks = "outside",
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black"),
                      yaxis = dict(tickformat = ".1%",
                                   ticks = "outside",
                                   title = "Pay Gap vs. " + controls[category]["omitted"][:16],
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black",
                                   showgrid=True,
                                   gridcolor="#e8e8e8",
                                   gridwidth=0.5,
                                   zeroline=True,
                                   zerolinecolor="#e8e8e8",
                                   zerolinewidth=2),
                      width=max(500, len(my_range) * 100),
                      margin=dict(r=0),
                      plot_bgcolor="white",
                      title_font = dict(size=20))

    if not os.path.exists("images"):
        os.mkdir("images")

    fig.write_image("images/" + category + "_" + exp + "_waterfall.jpg", scale=2)
    
    if online:
        py.plot(fig, filename=exp)

    fig.show()

def output_kde(df: pd.DataFrame, 
               category: str,
               exp: str,
               exp_list: list,
               online: bool=False) -> None:
    
    fig = ff.create_distplot([df[df[category] > -1][df[exp] == c][category] for c in exp_list], exp_list, bin_size=2, show_hist=True, show_rug=False)
    
    fig.update_layout(title = "Histogram: " + category,
                      xaxis = dict(tickangle = 0,
                                   ticks = "outside",
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black"),
                      yaxis = dict(tickformat = ".1%",
                                   ticks = "outside",
                                   showline=True,
                                   linewidth=0.5,
                                   linecolor="black",
                                   showgrid=True,
                                   gridcolor="#e8e8e8",
                                   gridwidth=0.5,
                                   zeroline=True,
                                   zerolinecolor="#e8e8e8",
                                   zerolinewidth=2),
                      legend = go.layout.Legend(x=0,
                                              y=1,
                                              yanchor="bottom",
                                              orientation="h"),
                      margin=dict(r=0),
                      plot_bgcolor="white",
                      title_font = dict(size=20))
    
    if not os.path.exists("images"):
        os.mkdir("images")

    fig.write_image("images/" + category + "_" + exp + "_kde.jpg", scale=2)
    
    if online:
        py.plot(fig, filename=exp + "_kde")
    
    fig.show()