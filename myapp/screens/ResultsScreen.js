import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { PieChart } from 'react-native-chart-kit';
import api from '../api';

export default function ResultsScreen({ route }) {
  const { userId } = route.params;
  const [results, setResults] = useState(null);

  useEffect(() => {
    const loadResults = async () => {
      try {
        const data = await api.getResults(userId);
        setResults(data);
      } catch (error) {
        console.error('Ошибка:', error);
      }
    };

    loadResults();
  }, []);

  if (!results) {
    return (
      <View style={styles.container}>
        <Text>Загрузка результатов...</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Ваши результаты</Text>
      
      <View style={styles.card}>
        <Text style={styles.score}>
          Общий уровень: {results.totalScore.toFixed(1)}/10
        </Text>
        <Text style={styles.interpretation}>
          {results.interpretation}
        </Text>
      </View>

      <View style={styles.chartContainer}>
        <PieChart
          data={[
            { name: 'Высокий', count: results.highCount, color: '#e74c3c' },
            { name: 'Средний', count: results.mediumCount, color: '#f39c12' },
            { name: 'Низкий', count: results.lowCount, color: '#2ecc71' },
          ]}
          width={300}
          height={200}
          chartConfig={{
            color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
          }}
          accessor="count"
          backgroundColor="transparent"
          paddingLeft="15"
        />
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>По категориям:</Text>
        {results.categories.map(category => (
          <View key={category.name} style={styles.category}>
            <Text>{category.name}: {category.score.toFixed(1)}</Text>
            <View style={styles.barContainer}>
              <View style={[
                styles.bar,
                { 
                  width: `${category.score * 10}%`,
                  backgroundColor: category.score > 7 ? '#e74c3c' : 
                                  category.score > 4 ? '#f39c12' : '#2ecc71'
                }
              ]} />
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  score: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  chartContainer: {
    alignItems: 'center',
    marginVertical: 20,
  },
  category: {
    marginBottom: 10,
  },
  barContainer: {
    height: 10,
    backgroundColor: '#ecf0f1',
    borderRadius: 5,
    marginTop: 5,
  },
  bar: {
    height: '100%',
    borderRadius: 5,
  },
});