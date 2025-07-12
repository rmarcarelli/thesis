"""
Base variable class for kinematic coordinate transformations.

This module defines the base Variable class that handles transformations between
different kinematic coordinate systems (e.g., log_gamma, eta, momentum, etc.)
with proper Jacobian tracking for integration purposes.
"""

########## VARIABLE CLASS ##########
# Could use autodifferentiation to avoid calculating the jacobian directly,
# although sometimes there are algebraic simplifications that autodiff can't find.

VARIABLE_REGISTRY = {}

class Variable():
    """
    Base class for kinematic variable transformations.
    
    This class provides a framework for transforming between different kinematic
    coordinate systems while properly tracking Jacobian determinants for integration.
    Each variable has a canonical form and can transform to/from other variables.
    
    Parameters
    ----------
    aliases : str or list of str
        String aliases for the variable (e.g., 'log_gamma', 'lngamma')
    to_canonical : function
        Function to transform from this variable to canonical form
    from_canonical : function
        Function to transform from canonical form to this variable
    canonical_var : Variable, optional
        The canonical variable (default: self)
    name : str, optional
        Display name for the variable (default: first alias)
    """

    def __init__(self, aliases, to_canonical, from_canonical, canonical_var = None, name = None):
        """
        Initialize a Variable instance.
        
        Parameters
        ----------
        aliases : str or list of str
            String aliases for the variable
        to_canonical : function
            Function to transform to canonical form
        from_canonical : function
            Function to transform from canonical form
        canonical_var : Variable, optional
            The canonical variable (default: self)
        name : str, optional
            Display name for the variable (default: first alias)
        """
        
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
        """
        Get variable from registry using bracket notation.
        
        Parameters
        ----------
        alias : str
            Variable alias to look up
            
        Returns
        -------
        Variable
            Variable instance from registry
        """
        return cls.from_registry(alias)

    def __call__(self, u, var = None, context = None, **kwargs):
        """
        Transform variable u from var to self without Jacobian.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Input variable type (default: canonical variable)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        float or array-like
            Transformed variable value
        """
        
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u = var.to_canonical(u, context = context, include_jacobian = False)
        return self.from_canonical(canonical_u, context = context, include_jacobian = False)
        
    def jacobian(self, u, var = None, context = None, **kwargs):
        """
        Compute Jacobian for transformation from var to self.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Input variable type (default: canonical variable)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        float or array-like
            Jacobian determinant
        """
        
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u, canonical_jacob = var.to_canonical(u, context = context, include_jacobian = True)
        _, jacob = self.from_canonical(canonical_u, context = context, include_jacobian = True)
        return jacob * canonical_jacob

    def transform(self, u, var = None, include_jacobian = True, context = None, **kwargs):
        """
        Transform variable u from var to self with optional Jacobian.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Input variable type (default: canonical variable)
        include_jacobian : bool, optional
            Whether to return Jacobian (default: True)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        tuple or float/array-like
            (transformed_value, jacobian) if include_jacobian=True,
            otherwise transformed_value
        """
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
        """
        Transform variable u from self to var without Jacobian.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Output variable type (default: canonical variable)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        float or array-like
            Transformed variable value
        """
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u = self.to_canonical(u, context = context, include_jacobian = False)
        return var.from_canonical(canonical_u, context = context, include_jacobian = False)

    def inverse_jacobian(self, u, var = None, context = None, **kwargs):
        """
        Compute Jacobian for transformation from self to var.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Output variable type (default: canonical variable)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        float or array-like
            Jacobian determinant
        """
        if not var:
            var = self.canonical_var
        var = Variable[var]
        
        context = context if context else kwargs
        
        canonical_u, canonical_jacob = self.to_canonical(u, context = context, include_jacobian = True)
        _, jacob = var.from_canonical(canonical_u, context = context, include_jacobian = True)
        return jacob * canonical_jacob

    def inverse_transform(self, u, var = None, include_jacobian = True, context = None, **kwargs):
        """
        Transform variable u from self to var with optional Jacobian.
        
        Parameters
        ----------
        u : float or array-like
            Input variable value
        var : Variable or str, optional
            Output variable type (default: canonical variable)
        include_jacobian : bool, optional
            Whether to return Jacobian (default: True)
        context : dict, optional
            Context for transformation
        **kwargs
            Additional context parameters
            
        Returns
        -------
        tuple or float/array-like
            (transformed_value, jacobian) if include_jacobian=True,
            otherwise transformed_value
        """
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
        """
        Get variable from registry by alias.
        
        Parameters
        ----------
        alias : str or Variable
            Variable alias or instance
            
        Returns
        -------
        Variable
            Variable instance from registry
            
        Raises
        ------
        KeyError
            If alias is not found in registry
        """
        
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
        """String representation of the variable."""
        return f'{self.name}({self.canonical_var.name})'

def IDENTITY(u, *, context = None, include_jacobian = True):
    """
    Identity transformation function.
    
    Parameters
    ----------
    u : float or array-like
        Input value
    context : dict, optional
        Context (unused)
    include_jacobian : bool, optional
        Whether to return Jacobian (default: True)
        
    Returns
    -------
    tuple or float/array-like
        (u, 1) if include_jacobian=True, otherwise u
    """
    if include_jacobian:
        return u, 1
    else:
        return u