#!/bin/bash
OUTPUT="aura_architecture.txt"
DESKTOP="/Users/bryan/Desktop/aura_architecture.txt"

echo "# Aura Full Architecture: Complete Integrated Mind" > $OUTPUT
echo "# Compiled: $(date)" >> $OUTPUT
echo "# Includes all core modules and subsystems." >> $OUTPUT

# Find all .py files in core/ recursively
FILES=$(find core -name "*.py" | sort)

for f in $FILES; do
    echo -e "\n\n================================================================================" >> $OUTPUT
    echo "FILE: $f" >> $OUTPUT
    echo -e "================================================================================\n" >> $OUTPUT
    cat "$f" >> $OUTPUT
done

# Ensure the file is on the Desktop and is the "most recent" 
# (touching it ensures the timestamp is now)
cp $OUTPUT "$DESKTOP"
touch "$DESKTOP"

echo "Full architecture compiled to $OUTPUT and copied to $DESKTOP"
