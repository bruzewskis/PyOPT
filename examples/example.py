import PyOPT as po

# Import and edit existing project
existing_project = po.projectFromFile('fakeproj.xml')

block = existing_project['A']['block1']

for scan in block:
    print(scan.type) #simpleScan vs scanLoop
    
    if isinstance(scan.type, po.simpleScan) and scan.duration < 30:
        scan.duration = 30
        scan.comments = 'Edited automatically'
        
existing_project.write('fakeproj2.xml', overwrite=True)

# Build new project from scratch
new_project = po.Project(name='00A-000')
new_project.meta.globalID = 8675
print(new_project.meta) # prints meta object in dict form

# Generate meta

# Generate sources

# Generate resources

# Adding a config
myconfig = po.newConfig(name='A') # manual creation
new_project.addConfig(myconfig)

print(new_project.configs) # prints list of config objects
print(myconfig) # prints meta info about particular config

# Adding blocks
myblock = po.blockFromFile('testblock.fits') # add from file
new_project.addBlock(myblock, config='A')
#new_project['A'].add_block(myblock) # Equivalent

# Show blocks in a config and rename one
print(new_project['A'].blocks) # will show blocks, including our 'testblock'
new_project['A']['testblock'].rename('block1')
print(new_project['A'].blocks)


