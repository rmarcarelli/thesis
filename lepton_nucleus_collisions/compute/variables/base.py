
########## VARIABLE CLASS ##########
# Could use autodifferentiation to avoid calculating the jacobian directly,
# although sometimes there are algebraic simplifications that autodiff can't find.

VARIABLE_REGISTRY = {}

class Variable():

    def __init__(self, aliases, to_canonical, from_canonical, canonical_var = None, name = None):
        
        if isinstance(aliases, str):
            aliases = [aliases]
            
        if not all(isinstance(alias, str) for alias in aliases):
            raise ValueError(f"All aliases must be strings. Got: {aliases}.")
                
        for idx in range(len(aliases)):
            aliases[idx] = aliases[idx].lower()
            for c in '()_ ':
                aliases[idx] = aliases[idx].replace(c, '')
        self.aliases = aliases
        self.name = name if name else aliases[0]
            
        self.to_canonical = to_canonical
        self.from_canonical = from_canonical
        
        self.canonical_var = canonical_var if canonical_var else self

        for alias in self.aliases:
            if alias in VARIABLE_REGISTRY:
                raise ValueError(f"Alias '{alias}' is already registered to another variable.")
            VARIABLE_REGISTRY[alias] = self
        VARIABLE_REGISTRY[self] = self


    def __class_getitem__(cls, alias):
        return cls.from_registry(alias)

    def __call__(self, u, var = None, context = None, **kwargs):
        
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u = var.to_canonical(u, context = context, include_jacobian = False)
        return self.from_canonical(canonical_u, context = context, include_jacobian = False)
        
    def jacobian(self, u, var = None, context = None, **kwargs):
        
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u, canonical_jacob = var.to_canonical(u, context = context, include_jacobian = True)
        _, jacob = self.from_canonical(canonical_u, context = context, include_jacobian = True)
        return jacob * canonical_jacob

    def transform(self, u, var = None, include_jacobian = True, context = None, **kwargs):
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs

        if include_jacobian:
            canonical_u, canonical_jacob = var.to_canonical(u, context = context, include_jacobian = True)
            out, jacob = self.from_canonical(canonical_u, context = context, include_jacobian = True)
            return out, jacob * canonical_jacob
        else:
            return self(u, var = var, context = context)

    def inverse(self, u, var = None, context = None, **kwargs):
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u = self.to_canonical(u, context = context, include_jacobian = False)
        return var.from_canonical(canonical_u, context = context, include_jacobian = False)

    def inverse_jacobian(self, u, var = None, context = None, **kwargs):
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u, canonical_jacob = self.to_canonical(u, context = context, include_jacobian = True)
        _, jacob = var.from_canonical(canonical_u, context = context, include_jacobian = True)
        return jacob * canonical_jacob

    def inverse_transform(self, u, var = None, include_jacobian = True, context = None, **kwargs):
        if not var:
            var = self.canonical_var
        var = Variable[var]

        context = context if context else kwargs

        if include_jacobian:
            canonical_u, canonical_jacob = self.to_canonical(u, context = context, include_jacobian = True)
            out, jacob = var.from_canonical(canonical_u, context = context, include_jacobian = True)
            return out, jacob * canonical_jacob
        else:
            return self(u, var = var, **kwargs)

    @staticmethod
    def from_registry(alias):
        
        if type(alias) == str:
            key = alias.lower()
            for c in '()_ ':
                key = key.replace(c, '')
        else:
            key = alias
            
        if key not in VARIABLE_REGISTRY:
            raise KeyError(f"No variable registered under alias '{alias}'.")
        return VARIABLE_REGISTRY[key]
    
    def __repr__(self):
        return f'{self.name}({self.canonical_var.name})'

def IDENTITY(u, *, context = None, include_jacobian = True):
    if include_jacobian:
        return u, 1
    else:
        return u