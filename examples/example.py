import PyOPT as po

# Import and edit existing project
existing_project = po.ProjectFromFile('fakeproj.xml')

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

# Generate sources
# default should be VLA source list
mysources = po.SourcesFromFile('targets.fits')
mysources.write_OPT('sources.pst')

# Generate resources
# default to VLA resources
mysetups = po.Resources()

# Adding a config
myconfig = po.Config(name='A') # manual creation
new_project.addConfig(myconfig)

print(new_project.configs) # prints list of config objects
print(myconfig) # prints meta info about particular config

# Adding blocks
myblock = po.Block('testblock', sources=mysources, resources=mysetups) # create empty block
new_project.add_block(myblock, config='A')
#new_project['A'].add_block(myblock) # Equivalent

# Show blocks in a config and rename one
print(new_project['A'].blocks) # will show blocks, including our 'testblock'
new_project['A']['testblock'].rename('block1')
print(new_project['A'].blocks)

# Build a new scan
cbandsetup = mysetups['C00f0']
myscan = po.Scan(target = mysources['J0000+0000'], setup = cbandsetup, dur=10)
somesources = [mysources['J0000+0000'], 
               mysources['J1111+1111'], 
               mysources['J2222+2222']]
somescans = po.Scan(target = somesources, setup = cbandsetup, dur=10) # returns list of scans
scanloop = po.Scan(target = somesources, setup = cbandsetup, repeat=10, dur=[10,9,8]) # returns scan loop

# Add to block
new_project['A']['block1'].add_scan(somescans)

# Validate
report = po.validate(new_project, show_plots=True) # Runs validator over project, returns results

# Write out
new_project.writeOPT('myproject.xml', overwrite=True)