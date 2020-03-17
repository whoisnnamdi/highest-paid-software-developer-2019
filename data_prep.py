import pandas as pd
import numpy as np

to_drop = ["Respondent", "OpenSource", "CareerSat", "JobSat", "JobSeek", "ResumeUpdate", "SurveyLength", "SurveyEase",
           "WelcomeChange", "EntTeams", "ScreenName", "LastIn", "SO", "BlockchainIs", "BlockchainOrg", "WorkChallenge", "BetterLife", "OffOn", "Currency",
           "CompTotal", "CompFreq", "MainBranch", "PlatformDesireNextYear", "LanguageDesireNextYear", "DatabaseDesireNextYear", 
           "MiscTechDesireNextYear", "WebFrameDesireNextYear", "MgrMoney", "ITperson", "Age1stCode", "MgrIdiot", "MgrWant", "LastHireDate", 
           "Containers", "WorkLoc", "SONewContent"]

to_keep = ['Hobbyist', 'OpenSourcer', 'Employment', 'Country', 'Student', 'EdLevel', 'UndergradMajor', 'EduOther', 'OrgSize',
           'DevType', 'YearsCode', 'YearsCodePro', 'FizzBuzz', 'JobFactors', 'CurrencySymbol', 'CurrencyDesc',
           'ConvertedComp', 'WorkWeekHrs', 'WorkPlan', 'WorkRemote', 'ImpSyn', 'CodeRevHrs', 'UnitTests',
           'PurchaseHow', 'PurchaseWhat', 'LanguageWorkedWith', 'DatabaseWorkedWith', 'PlatformWorkedWith', 'WebFrameWorkedWith',
           'MiscTechWorkedWith', 'DevEnviron', 'OpSys', 'SocialMedia', 'Extraversion', 'Age', 'Gender', 'Trans', 'Sexuality', 
           'Ethnicity', 'Dependents']

num_columns = ["Age", "YearsCode", "YearsCodePro", "WorkWeekHrs", "CodeRevHrs"]

def misc_prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Misc data cleaning and prep
    """
    
    # Only consider those with income between $10,000 and $250,000
    df = df[(df["ConvertedComp"] >= 10000) & (df["ConvertedComp"] <= 250000)]
    df["ConvertedComp"] = np.log(df["ConvertedComp"])

    # Only consider US respondents
    df = df[df["Country"] == "United States"]
    #df.drop(["Country"], axis=1, inplace=True)

    # Only consider 18+ respondents
    df = df[df["Age"] >= 21]

    # Only consider respondents in the workforce
    df = df[df["Employment"] != "Retired"]
    df = df[df["Employment"] != "Not employed, and not looking for work"]

    df = df[df["WorkWeekHrs"] >= 5]

    # No students
    df = df[["Student" not in str(x) for x in df["DevType"].values]]
    df = df[df["Student"] == "No"]

    # Only consider those with at least some education
    df = df[df["EdLevel"] != "I never completed any formal education"]

    df = df.fillna("no_answer")

    num_columns = ["Age", "YearsCode", "YearsCodePro", "WorkWeekHrs", "CodeRevHrs"]

    # Convert numeric columns to int
    for col in num_columns:
        df[col] = df[col].astype("int32", errors="ignore")

    df["YearsCode"].replace("Less than 1 year", "0", inplace=True)
    df["YearsCode"].replace("More than 50 years", "51", inplace=True)

    # Exclude respondents who selected multiple gender, race, or sexual orientation options
    df = df[~df["Gender"].str.contains(";")]
    df = df[~df["Ethnicity"].str.contains(";")]
    df = df[~df["Sexuality"].str.contains(";")]

    # Keep certain columns
    df = df[to_keep]

    df = df.rename(columns = {"ConvertedComp": "Income"})

    for col in num_columns:
        df[col] = df[col].apply(string_replace)

    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def col_drop(df: pd.DataFrame, to_drop: list) -> pd.DataFrame:

    df_dropped = df.copy()

    for flag in to_drop:
        try:
            df_dropped.drop([x for x in df_dropped.columns if flag == x], axis=1, inplace=True)
        except:
            pass

    return df_dropped

def string_replace(s: str) -> int:
    try:
        s = float(s)
    except:
        s = -1000

    return s

def text_clean(text: str) -> str:
    text = str(text).replace(" ", "_").replace("-", "_").replace(
        ",", "_").replace(".", "").replace("+", "p").replace("#", "s").replace(
            "/", "_").replace("'", "").replace("ʼ", "").replace(
                "(", "_").replace(")", "_").replace("’", "").replace(
                    "__", "_").replace("__", "_").replace("“", "").replace(
                        "”", "").replace(":", "_").replace("&", "_").lower()

    banned = ["participated_in_", "_or", "_of", "_etc", "_employees", "taken_", 
              "_african_descent", "employed_", "developer_", "specialist_",
              "_european_descent", "contributed_to_", "completed_an_", 
              "a_full_time_training_program_", "in_person_", "received_", 
              "online_coding_", "_eg_hackerrank_codechef_topcoder_",
              "_eg_american_high_school_german_realschule_gymnasium_"]

    for t in banned: text = text.replace(t, "")

    return text

def create_controls(df: pd.DataFrame, exclude: str) -> dict:

    df = df.copy()
    
    controls = {}

    for col in df.columns:
        if col != exclude:
            controls[col] = {
                "omitted": text_clean(pd.Series([x for sub in list(
                           df[col]
                           .apply(text_clean)
                           .apply(lambda x: str(x).split(";"))) for x in sub])
                           .value_counts()
                           .idxmax()), 
                "controls": list(set([x for sub in list(
                            df[col]
                            .apply(text_clean)
                            .apply(lambda x: str(x).split(";"))) for x in sub]))}

    return controls

def design_matrix(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    dm = df.copy()
    
    for control in controls.keys():
        dm[control] = dm[control].apply(text_clean)

        if control in num_columns:
            for c in controls[control]["controls"]:
                dm[control+"_"+c] = (dm[control] == c) * 1

        else:
            for c in controls[control]["controls"]:
                dm[control+"_"+c] = dm[control].apply(lambda x: c in str(x).split(";")) * 1

        dm.drop(control, axis=1, inplace=True)
        dm.drop(control+"_"+controls[control]["omitted"], axis=1, inplace=True)

    dm = dm[sorted(dm.columns)]

    return dm

def bucketize(df: pd.DataFrame):
    """
    Bucket numerical columns
    """
    
    df = df.copy()
    
    num_labels = [
        ["no_answer", "21-25", "26-30", "31-35", "35-40", "41-45", "45-50", "51-55", "55-60", "61-65", "66-"],
        ["no_answer","00-05", "06-10", "11-15", "16-20", "21-25", "26-30", "31-35", "35-40", "41-"],
        ["no_answer","00-05", "06-10", "11-15", "16-20", "21-25", "26-30", "31-35", "35-40", "41-"],
        ["no_answer","00-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-"],
        ["no_answer","01-02", "02-04", "04-06", "06-08", "08-10", "11-15", "16-20", "21-"]
    ]

    num_buckets = [
        np.array([-1001,0,25,30,35,40,45,50,55,60,65,100]),
        np.array([-1001,0,5,10,15,20,25,30,35,40,100]),
        np.array([-1001,0,5,10,15,20,25,30,35,40,100]),
        np.array([-1001,0,10,20,30,40,50,60,70,80,200]),
        np.array([-1001,0,2,4,6,8,10,15,20,200])
    ]
    
    for i, col in enumerate(num_columns):
        df[col] = pd.cut(df[col], 
                         num_buckets[i], 
                         labels=num_labels[i]).astype("str")
        
    return df

def mean_replace(df: pd.DataFrame,
                 controls: dict,
                 col_check: str,
                 cat_replace: str) -> pd.DataFrame:
    """
    Replaces columns matching cat_replace for rows where col_check == 1 with mean values where col_check == 0
    """
    
    df = df.copy()
    
    to_replace = [col for col in df.columns if cat_replace in col]
    
    means = df[df[col_check] == 0][to_replace].mean()
    
    rows = df[col_check] == 1
    
    for col in to_replace:
        df.loc[df[rows].index, col] = means[col]
    
    df.drop(col_check, axis=1, inplace=True)
    
    controls[cat_replace]["controls"].remove(col_check.replace(cat_replace + "_",""))
    
    return df, controls#[[col for col in df.columns if cat_replace in col]]

def prune_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prunes dataframe columns to only include those with "category_" in the name
    
    Returns: Pruned dataframe
    """

    matching = [x for x in df.columns if category+"_" in x]

    return df[matching].copy()