from state import State, Todos


class TestTodos:
    def test_add_items(self):
        todos = Todos()
        added = todos.add(["a", "b", "c"])
        assert added == ["a", "b", "c"]
        assert str(todos) == "- [ ] a\n- [ ] b\n- [ ] c"

    def test_add_strips_whitespace(self):
        todos = Todos()
        added = todos.add(["  a  ", "b "])
        assert added == ["a", "b"]

    def test_add_deduplicates(self):
        todos = Todos()
        todos.add(["a", "b"])
        added = todos.add(["b", "c"])
        assert added == ["c"]
        assert str(todos) == "- [ ] a\n- [ ] b\n- [ ] c"

    def test_remove_items(self):
        todos = Todos()
        todos.add(["a", "b", "c"])
        removed, not_exists = todos.remove(["a", "c"])
        assert set(removed) == {"a", "c"}
        assert not_exists == []
        assert str(todos) == "- [ ] b"

    def test_remove_nonexistent(self):
        todos = Todos()
        todos.add(["a"])
        removed, not_exists = todos.remove(["a", "x"])
        assert removed == ["a"]
        assert not_exists == ["x"]

    def test_complete_items(self):
        todos = Todos()
        todos.add(["a", "b", "c"])
        completed, not_exists = todos.complete(["a", "c"])
        assert set(completed) == {"a", "c"}
        assert not_exists == []

    def test_complete_nonexistent(self):
        todos = Todos()
        todos.add(["a"])
        completed, not_exists = todos.complete(["a", "x"])
        assert completed == ["a"]
        assert not_exists == ["x"]

    def test_complete_is_idempotent(self):
        todos = Todos()
        todos.add(["a", "b"])
        todos.complete(["a"])
        completed, not_exists = todos.complete(["a"])
        assert completed == []
        assert not_exists == []

    def test_complete_strips_whitespace(self):
        todos = Todos()
        todos.add(["a"])
        completed, not_exists = todos.complete(["  a  "])
        assert completed == ["a"]
        assert not_exists == []

    def test_remove_strips_whitespace(self):
        todos = Todos()
        todos.add(["a"])
        removed, not_exists = todos.remove(["  a  "])
        assert set(removed) == {"a"}
        assert not_exists == []


class TestTodosStr:
    def test_empty(self):
        todos = Todos()
        assert str(todos) == "No todos."

    def test_with_items(self):
        todos = Todos()
        todos.add(["a", "b"])
        assert str(todos) == "- [ ] a\n- [ ] b"

    def test_with_completed(self):
        todos = Todos()
        todos.add(["a", "b", "c"])
        todos.complete(["b"])
        assert str(todos) == "- [ ] a\n- [x] b\n- [ ] c"


class TestState:
    def test_defaults(self):
        state = State()
        assert state.iterations == 0
        assert str(state.todos) == "No todos."
