
def stretch_string(input_string, h, l):
    words = input_string.split()
    total_length = sum(len(word) for word in words)
    total_area = h * l
    space_needed = total_area - total_length
    spaces_between_words = len(words) - 1

    # The following lines are affected by this conditional statement
    if spaces_between_words == 0:
        # This line is affected by the conditional statement
        stretched_string = words[0] + ' ' * space_needed
        # This line is affected by the conditional statement
        if(len(stretch_string) >= l):
            # This line is affected by the conditional statement
            return stretched_string[:l] + '\n' + stretched_string[l:]
        else:
            # This line is affected by the conditional statement
            return stretched_string
    
    base_space, extra_spaces = divmod(space_needed, spaces_between_words)

    stretched_words = []
    '''
    This loop iterates through the words and adds spaces between them.
    '''
    for i, word in enumerate(words[:-1]):
        stretched_words.append(word)
        # This line is affected by the conditional statement within the loop
        stretched_words.append(' ' * (base_space + (1 if i < extra_spaces else 0)))
    stretched_words.append(words[-1])
    
    stretched_string = ''.join(stretched_words)

    '''
    This loop creates a list of lines, each line with a maximum length of l
    '''
    lines = [stretched_string[i:i+l] for i in range(0, len(stretched_string), l)] 

    '''
    This loop adds empty lines until the desired height is reached
    '''
    while len(lines) < h:
        lines.append(' ' * l)

    return '\n'.join(lines)