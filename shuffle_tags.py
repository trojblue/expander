import random
from tqdm.auto import tqdm
def shuffle_tags(tags, prob, place):
    # Split the tags into a list
    tag_list = tags.split(', ')

    # Shuffle each tag with probability prob
    for i in range(len(tag_list)):
        if random.random() < prob:
            # Choose a random distance to shuffle
            shuffle_dist = random.randint(1, place)

            # Calculate the new index for the tag
            new_index = min(max(0, i + random.randint(-shuffle_dist, shuffle_dist)), len(tag_list) - 1)

            # Swap the tags
            tag_list[i], tag_list[new_index] = tag_list[new_index], tag_list[i]

    # Join the tags back into a comma-separated string
    return ', '.join(tag_list)


if __name__ == '__main__':
    og_lines = []
    new_lines = []
    # Open the input and output files
    with open('tag_only.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
        # Read each line of the input file
        for line in tqdm(input_file):
            # Shuffle the tags in the line with a probability of 0.5 and a maximum shuffle distance of 2
            og_lines.append(line)
            new_lines.append(shuffle_tags(line.strip(), 0.2, 3))
            new_lines.append(shuffle_tags(line.strip(), 0.5, 2))
            new_lines.append(shuffle_tags(line.strip(), 0.5, 1))
            new_lines.append(shuffle_tags(line.strip(), 0.1, 3))

            # # Write the shuffled line to the output file
            # output_file.write(shuffled_line + '\n')
    all_lines = og_lines + new_lines
    random.shuffle(all_lines)

    with open("shuffled_tags.txt", 'w') as f:
        for item in all_lines:
            f.write("%s\n" % item)

    print("D")