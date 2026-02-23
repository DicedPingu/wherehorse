grid = ''

for a in range(0,8):
    grid += '['+','.join([str(b*3+a**2) for b in range(0,8)])+('],\n' if a < 7 else ']')

with open('testmap.txt', 'w') as g:
    g.write(grid)

print(grid)