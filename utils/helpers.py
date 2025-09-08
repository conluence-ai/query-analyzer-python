def toCamelCase(snake_str: str) -> str:
    """
    Convert a snake_case string to camelCase.
    
    Args:
        snake_str (str): The input string in snake_case format.
        
    Returns:
        str: The converted string in camelCase format.
    """
    components = snake_str.lower().split(' ')
    return components[0] + ''.join(x.title() for x in components[1:]) if components else ''