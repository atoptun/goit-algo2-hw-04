from typing import Optional


class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.value: Optional[int] = None


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def put(self, key: str, value: Optional[int] = None) -> None:
        if not isinstance(key, str) or not key:
            raise TypeError(f"Illegal argument for put: key = {key} must be a non-empty string")

        current = self.root
        for char in key:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        if current.value is None:
            self._size += 1
        current.value = value    

    def append(self, keys: list[str]) -> list[str]:
        errors = []
        for key in keys:
            try:
                self.put(key, self._size + 1)
            except TypeError as e:
                errors.append(e)
        return errors

    def get(self, key: str) -> Optional[int]:
        if not isinstance(key, str) or not key:
            raise TypeError(f"Illegal argument for get: key = {key} must be a non-empty string")

        current = self.root
        for char in key:
            if char not in current.children:
                return None
            current = current.children[char]
        return current.value

    def delete(self, key: str) -> bool:
        if not isinstance(key, str) or not key:
            raise TypeError(f"Illegal argument for delete: key = {key} must be a non-empty string")

        size_tmp = self._size

        def _delete(node: TrieNode, key: str, depth: int) -> bool:
            if depth == len(key):
                if node.value is not None:
                    node.value = None
                    self._size -= 1
                    return len(node.children) == 0
                return False

            char = key[depth]
            if char in node.children:
                should_delete = _delete(node.children[char], key, depth + 1)
                if should_delete:
                    del node.children[char]
                    return len(node.children) == 0 and node.value is None
            return False

        _delete(self.root, key, 0)
        return size_tmp != self._size

    def contains(self, key: str) -> bool:
        if not isinstance(key, str) or not key:
            raise TypeError(f"Illegal argument for contains: key = {key} must be a non-empty string")
        
        return self.get(key) is not None

    def longest_prefix_of(self, s: str) -> str:
        if not isinstance(s, str) or not s:
            raise TypeError(f"Illegal argument for longestPrefixOf: s = {s} must be a non-empty string")

        current = self.root
        longest_prefix = ""
        current_prefix = ""
        for char in s:
            if char in current.children:
                current = current.children[char]
                current_prefix += char
                if current.value is not None:
                    longest_prefix = current_prefix
            else:
                break
        return longest_prefix

    def keys_with_prefix(self, prefix: str) -> list[str]:
        if not isinstance(prefix, str):
            raise TypeError(f"Illegal argument for keysWithPrefix: prefix = {prefix} must be a string")

        current = self.root
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]

        result = []
        self._collect(current, list(prefix), result)
        return result

    def _collect(self, node: TrieNode, path: list[str], result: list[str]) -> None:
        if node.value is not None:
            result.append("".join(path))
        for char, next_node in node.children.items():
            path.append(char)
            self._collect(next_node, path, result)
            path.pop()

    def keys(self) -> list[str]:
        result = []
        self._collect(self.root, [], result)
        return result

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size
    

def test():
    trie = Trie()
    trie.put("apple", 1)
    trie.put("app", 2)
    trie.put("banana", 3)
    trie.put("band", 4)
    trie.put("bandana", 5)

    assert trie._size == 5

    assert trie.get("apple") == 1
    assert trie.get("app") == 2
    assert trie.get("banana") == 3
    assert trie.get("band") == 4
    assert trie.contains("ban") == False
    assert trie.contains("band") == True
    assert trie.get("bandana") == 5
    assert trie.contains("bandage") == False
    assert trie.contains("apricot") == False

    trie.append(["cat", "caterpillar", "dog"])
    assert trie._size == 8
    assert trie.get("cat") == 6
    assert trie.get("caterpillar") == 7
    assert trie.get("dog") == 8

    assert trie.longest_prefix_of("bandage") == "band"
    assert trie.longest_prefix_of("apricot") == ""
    assert trie.longest_prefix_of("bat") == ""

    assert sorted(trie.keys_with_prefix("ba")) == ["banana", "band", "bandana"]
    assert sorted(trie.keys_with_prefix("ban")) == ["banana", "band", "bandana"]
    assert sorted(trie.keys_with_prefix("band")) == ["band", "bandana"]
    assert trie.keys_with_prefix("bandana") == ["bandana"]
    assert trie.keys_with_prefix("x") == []
    assert trie.keys() == ["app", "apple", "banana", "band", "bandana", "cat", "caterpillar", "dog"]

    assert trie.delete("band") == True
    assert trie.delete("band") == False
    assert trie.contains("band") == False
    assert trie.contains("bandana") == True
    assert trie._size == 7
    assert trie.delete("apple") == True
    assert trie.contains("apple") == False
    assert trie._size == 6

    print("All tests passed!")


def test_2():
    trie = Trie()
    words = ["apple", "application", "appetizer", "banana", "band", "banner", "ball", "bat", "battery"]

    trie.append(words)
    # # Додаємо слова до Trie
    # for index, word in enumerate(words):
    #     trie.put(word, index)
    assert len(trie) == len(words)
    assert trie.get("apple") == 1

    # Функція автозаповнення
    def autocomplete(trie, prefix):
        return trie.keys_with_prefix(prefix)

    # Приклад використання автозаповнення
    user_input = "app"
    suggestions = autocomplete(trie, user_input)
    assert sorted(suggestions) == sorted(["apple", "appetizer", "application"])
    print(f"Пропозиції для '{user_input}': {suggestions}")
    



if __name__ == "__main__":
    test()
    test_2()
    