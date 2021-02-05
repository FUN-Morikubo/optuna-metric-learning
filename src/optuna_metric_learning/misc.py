class ParameterGenerator():
    def __init__(self, trial, fix_params, logger):
        self.trial = trial
        self.fix_params = fix_params

        self.logger = logger
    
    def __getattr__(self, func_name): 
        if func_name.startswith("suggest"): 
            def _tmp(name, *args): 
                if name in self.fix_params: 
                    val = self.fix_params[name]
                    self.logger.critical(f"Fix parameter for {name}: {val}")
                    return val
                else: 
                    try:
                        val = getattr(self.trial, func_name)(name, *args) 
                    except Exception as err:
                        print(f"Error while loading parameter: {name}")
                        raise err
                    self.logger.critical(f"Sample parameter for {name}: {val}")
                    return val
            return _tmp 
        else:
            getattr(super(), func_name)