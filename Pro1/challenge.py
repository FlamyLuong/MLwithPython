def is_isomorphic (s:str , t:str) -> bool:
    mapping = {}
    mapped_before = {}
    equivalent = True

    for i in range(len(s)):
        if s[i] in mapping:
            mapped = mapping[s[i]]

            if mapped != t[i]:
                equivalent = False
                break
        else:
            if t[i] in mapped_before:
                equivalent = False
                break
            else:
                mapped_before[t[i]] = True
                mapping[s[i]] = t[i]

    return equivalent

print(is_isomorphic("badc", "baba"))