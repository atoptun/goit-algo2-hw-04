from typing import Optional
from trie import Trie, TrieNode


class Homework(Trie):

    def __init__(self):
        super().__init__()
        self._reversed_trie = Trie()

    def put(self, key: str, value: Optional[int] = None) -> None:
        super().put(key, value)
        self._reversed_trie.put(key[::-1], value)

    def delete(self, key: str) -> bool:
        deleted = super().delete(key)
        if deleted:
            self._reversed_trie.delete(key[::-1])
        return deleted

    def keys_with_suffix(self, suffix: str) -> list[str]:
        """
        Retrieve all keys in the trie that end with the given suffix.

        Args:
            suffix (str): Suffix pattern to search for.
        
        Returns:
            list[str]: List of keys ending with the given suffix.
        """
        reversed_result = self._reversed_trie.keys_with_prefix(suffix[::-1])
        result = [key[::-1] for key in reversed_result]
        return result

    def count_words_with_suffix(self, pattern: str) -> int:
        """
        Count the number of words in the trie that end with the given suffix.

        Args:
            pattern (str): Suffix pattern to search for.
        
        Returns:
            int: Number of words ending with the given suffix.
        """
        if not isinstance(pattern, str) or not pattern:
            raise TypeError(f"Illegal argument for count_words_with_suffix: pattern = {pattern} must be a non-empty string")

        reversed_pattern = pattern[::-1]
        current = self._reversed_trie.root
        for char in reversed_pattern:
            if char not in current.children:
                return 0
            current = current.children[char]
        
        def _count_words(node: TrieNode) -> int:
            count = 1 if node.value is not None else 0
            for child in node.children.values():
                count += _count_words(child)
            return count
        
        return _count_words(current)

    def has_prefix(self, prefix: str) -> bool:
        """
        Check if there is any word in the trie that starts with the given prefix.

        Args:
            prefix (str): Prefix to search for.
        
        Returns:
            bool: True if there is at least one word with the given prefix, False otherwise.
        """
        if not isinstance(prefix, str) or not prefix:
            raise TypeError(f"Illegal argument for has_prefix: prefix = {prefix} must be a non-empty string")
        
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


def test():
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    # for i, word in enumerate(words):
    #     trie.put(word, i)
    trie.append(words)

    print('Inserted words:', words)
    print('-' * 40)

    # Перевірка кількості слів, що закінчуються на заданий суфікс
    result = trie.count_words_with_suffix("e")
    print('Count words with suffix "e":', result)
    assert result == 1  # apple
    result = trie.count_words_with_suffix("ion")
    print('Count words with suffix "ion":', result)
    assert result == 1  # application
    result = trie.count_words_with_suffix("a")
    print('Count words with suffix "a":', result)
    assert result == 1  # banana
    result = trie.count_words_with_suffix("at")
    print('Count words with suffix "at":', result)
    assert result == 1  # cat
    result = trie.count_words_with_suffix("z")
    print('Count words with suffix "z":', result)
    assert result == 0  # no words

    print('-' * 40)

    # Перевірка отримання ключів за суфіксом
    result = trie.keys_with_suffix("e")
    print('Words with suffix "e":', result)
    assert set(result) == {"apple"}
    result = trie.keys_with_suffix("ion")
    print('Words with suffix "ion":', result)
    assert set(result) == {"application"}
    result = trie.keys_with_suffix("a")
    print('Words with suffix "a":', result)
    assert set(result) == {"banana"}
    result = trie.keys_with_suffix("at")
    print('Words with suffix "at":', result)
    assert set(result) == {"cat"}
    result = trie.keys_with_suffix("z")
    print('Words with suffix "z":', result)
    assert set(result) == set()  # no words

    print('-' * 40)

    # Перевірка наявності префікса
    result = trie.has_prefix("app")
    print('Has prefix "app":', result)
    assert result == True  # apple, application
    result = trie.has_prefix("bat")
    print('Has prefix "bat":', result)
    assert result == False
    result = trie.has_prefix("ban")
    print('Has prefix "ban":', result)
    assert result == True  # banana
    result = trie.has_prefix("ca")
    print('Has prefix "ca":', result)
    assert result == True  # cat

    print('-' * 40)

    print("All tests passed.")


if __name__ == "__main__":
    test()
