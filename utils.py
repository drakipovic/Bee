def create_accuracies_markdown_table(accuracies, n_trees):
    markdown = '| Variance Threshold '
    for nt in n_trees:
        markdown += '| {} trees '.format(nt)
    
    markdown += '\n'
    
    markdown += '| :---: '
    for nt in n_trees:
        markdown += '| --- '
    
    markdown += '\n'

    for vt in sorted(accuracies.keys()):
        markdown += '| **{}** '.format(vt)
        for acc, t in accuracies[vt]:
            markdown += '| {:.3f}% {:.1f}s '.format(float(acc), float(t))
        
        markdown += '\n'
    
    return markdown