# configs_gw.py

ALL_COVARIATES_NON_AR = [
    "vp", "impvar", "vrp", "lzrt", "ogap", "wtexas", "sntm", "ndrbl", 
    "skvw", "tail", "fbm", "dtoy", "dtoat", "ygap", "rdsp", "rsvix", 
    "tchi", "avgcor", "shtint", "disag", "ntis", "tbl", "d/p", "d/y", 
    "e/p", "d/e", "svar", "lty", "ltr", "tms", "dfy", "dfr", "infl", "b/m"
]

ALL_COVARIATES_AR = ALL_COVARIATES_NON_AR + ["equity_premium"]

MINIMAL_SETTING = ["tchi", "lty", "wtexas"]

WITHOUT_LOOKAHEAD = [
    "vp", "impvar", "vrp", "lzrt", 
    #"ogap", 
    "wtexas", 
    #"sntm", 
    "ndrbl", 
    "skvw", "tail", 
    #"fbm",
    "dtoy", "dtoat", "ygap", "rdsp", 
    #"rsvix", 
    #"tchi", 
    "avgcor", 
    #"shtint",
    "disag", "ntis", "tbl", "d/p", "d/y", 
    "e/p", "d/e", "svar", "lty", "ltr", "tms", "dfy", "dfr", "infl", "b/m"
]

WITHOUT_LOOKAHEAD_AR = WITHOUT_LOOKAHEAD + ["equity_premium"]

COMPLETED = [
    "lzrt", "wtexas", "skvw", "tail", "dtoy", "dtoat", "rdsp", "avgcor",
    "ntis", "tbl", "d/p", "d/y", "e/p", "d/e", "svar", "lty", "ltr",
    "tms", "dfy", "dfr", "infl", "b/m",
]

ALL = ALL_COVARIATES_NON_AR